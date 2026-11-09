"""Weather T720: c * Huber(delta=1), c=1,2,3,4,5; seed2026."""
import importlib.util
import io
import math
import re
import sys
import time
import traceback
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
CODE = HERE.parent.parent
DATA = Path(r'D:\all_datasets')
HORIZONS = [96, 192, 336, 720]
SEEDS = [2026, 2027, 2028]
RATES = [0., .05, .10, .15, .20]


def source_for(dataset):
    spec = importlib.util.spec_from_file_location('extra_' + dataset, CODE / (dataset + '.py'))
    source = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(source)
    return source


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    squared = absolute = 0.
    count = 0
    for x, history, future, target in loader:
        prediction = model(x.cuda(), history.cuda(), future.cuda())
        error = (prediction.cpu() - target).numpy().astype(np.float64)
        squared += np.square(error).sum()
        absolute += np.abs(error).sum()
        count += error.size
    return squared / count, absolute / count


def run(dataset, objective, delta, horizon, seed, coefficient):
    rate = 0.
    print("COEFFICIENT c={} | objective=c*Huber(delta=1); no added MAE".format(coefficient), flush=True)
    source = source_for(dataset)
    source.set_seed(seed)
    is_weather = dataset == 'weather'
    width, dropout = (128, .05) if is_weather else (16, .1)
    lr, decay, max_epochs, min_lr = (5e-4, 1e-4, 100, 1e-6) if is_weather else (2e-5, 1e-3, 1000, 1e-8)
    filename = 'weather/weather.csv' if is_weather else 'ETT-small/ETTm1.csv'
    with redirect_stdout(io.StringIO()):
        original_train, validation, test, variables, features = source.load_and_split_data(
            str(DATA / filename), seq_len=96, pred_len=horizon, batch_size=256, num_workers=0)
    actual_rate = 0.
    train = original_train
    train.sampler.generator = torch.Generator().manual_seed(seed)
    for i, loader in enumerate([train, validation, test]):
        loader.generator = torch.Generator().manual_seed(seed + 10000 * (i + 1))
    model = source.CTIAODEForecaster(96, horizon, variables, features,
        d_model=width, primary_idx=-1, dropout=dropout).cuda()
    model.stable_block.max_loops = 2
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=.5, patience=3, min_lr=min_lr)
    print('START dataset={} objective={} delta={} T={} seed={} eta={:.2f} actual_eta={:.6f} | {}'.format(
        dataset, objective, delta, horizon, seed, rate, actual_rate, datetime.now().isoformat()), flush=True)
    print('CONFIG width={} batch=256 dropout={} stable_max_loops=2 lr={} wd={} max_epochs={} early_stop=12'.format(
        width, dropout, lr, decay, max_epochs), flush=True)
    best, best_state, best_epoch, stale = float('inf'), None, 0, 0
    steps = over = 0
    max_norm = 0.
    failed = False
    started = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        for x, history, future, target in train:
            x, history, future, target = [v.cuda() for v in (x, history, future, target)]
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x, history, future)
            if objective == 'MSE':
                loss = F.mse_loss(prediction, target)
            elif objective == 'MAE':
                loss = F.l1_loss(prediction, target)
            else:
                loss = coefficient * F.huber_loss(prediction, target, delta=delta)
                if objective == 'ORHL':
                    loss = loss + 5. * F.l1_loss(prediction, target)
            if not torch.isfinite(loss).item():
                failed = True
                print('NONFINITE objective at epoch={}'.format(epoch), flush=True)
                break
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.).item()
            if not math.isfinite(norm):
                failed = True
                print('NONFINITE gradient at epoch={}'.format(epoch), flush=True)
                break
            steps += 1
            over += int(norm > 5.)
            max_norm = max(max_norm, norm)
            optimizer.step()
        if failed:
            break
        val_mse, val_mae = evaluate(model, validation)
        if not (math.isfinite(val_mse) and math.isfinite(val_mae)):
            failed = True
            print('NONFINITE validation at epoch={}'.format(epoch), flush=True)
            break
        scheduler.step(val_mse)
        if val_mse < best:
            best, best_epoch, stale = val_mse, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        print('EPOCH={} objective={} delta={} T={} seed={} eta={:.2f} val_MSE={:.9f} val_MAE={:.9f} ES={}/12 seconds={:.2f}'.format(
            epoch, objective, delta, horizon, seed, rate, val_mse, val_mae, stale, time.perf_counter()-epoch_start), flush=True)
        if stale >= 12:
            break
    if failed:
        mse = mae = float('nan')
    else:
        model.load_state_dict(best_state)
        mse, mae = evaluate(model, test)
    print('STABILITY steps={} preclip_grad_max={:.6f} clipping_fraction={:.6f} nonfinite_event={}'.format(
        steps, max_norm, over/max(steps, 1), int(failed)), flush=True)
    print('RESULT objective={} delta={} T={} seed={} eta={:.2f} | MSE={:.9f} MAE={:.9f} best_val_MSE={:.9f} best_epoch={} epochs={} seconds={:.2f} status={}'.format(
        objective, delta, horizon, seed, rate, mse, mae, best, best_epoch, epoch,
        time.perf_counter()-started, 'FAILED' if failed else 'OK'), flush=True)
    del model, optimizer, scheduler, best_state, prediction, loss
    torch.cuda.empty_cache()
    return mse, mae



def main():
    sys.dont_write_bytecode = True
    torch.set_num_threads(2)
    print('Huber multiplier experiment | Started: '+datetime.now().isoformat(), flush=True)
    print('Weather L=96 T=720 seed=2026 | c=1,2,3,4,5; delta fixed at 1; no horizon averaging.', flush=True)
    print('Each coefficient trains independently from the same initial weights and loader seeds. No L1 addition or contamination.', flush=True)
    print('D=128 batch=256 dropout=.05 stable_max_loops=2 AdamW lr=5e-4 wd=1e-4 norm5 clipping.', flush=True)
    print('Max epochs=100 early_stop=12; scheduler factor=.5 patience=3 min_lr=1e-6; best validation MSE selects weights.', flush=True)
    print('Scaling the objective can change optimizer/clipping behavior. Concurrent noise experiment may affect elapsed time.', flush=True)
    results=[]
    for coefficient in [1,2,3,4,5]:
        mse,mae=run('weather','Huber',1.,720,2026,coefficient)
        results.append((coefficient,mse,mae))
        print('COEFFICIENT_RESULT c={} | MSE={:.9f} MAE={:.9f}'.format(coefficient,mse,mae), flush=True)
    print('FINAL WEATHER T720 RESULTS', flush=True)
    for coefficient,mse,mae in results:
        print('c={} | MSE={:.9f} MAE={:.9f}'.format(coefficient,mse,mae), flush=True)
    print('Finished: '+datetime.now().isoformat(), flush=True)


if __name__=='__main__':
    with (HERE/'huber_coefficients.log').open('x',encoding='utf-8',buffering=1) as output:
        with redirect_stdout(output),redirect_stderr(output):
            try:
                main()
            except BaseException:
                traceback.print_exc()
                raise
