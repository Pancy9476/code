"""Weather: ORHL/MSE/MAE x no clipping/norm-5, seed 2026, four-horizon Avg."""
import importlib.util
import math
import sys
import time
import traceback
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
CODE = HERE.parent.parent
DATA = Path(r'D:\all_datasets\weather\weather.csv')
SEED = 2026
HORIZONS = [96, 192, 336, 720]


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    squared = absolute = 0.
    count = 0
    for x, history, future, target in loader:
        prediction = model(x.cuda(), history.cuda(), future.cuda())
        residual = (prediction.cpu() - target).numpy().astype(np.float64)
        squared += np.square(residual).sum()
        absolute += np.abs(residual).sum()
        count += target.numel()
    return squared / count, absolute / count


def run(source, objective, clipping, horizon):
    source.set_seed(SEED)
    train, validation, test, variables, features = source.load_and_split_data(
        str(DATA), seq_len=96, pred_len=horizon, batch_size=256, num_workers=0)
    train.sampler.generator = torch.Generator().manual_seed(SEED)
    for i, loader in enumerate([train, validation, test]):
        loader.generator = torch.Generator().manual_seed(SEED + 10000 * (i + 1))
    model = source.CTIAODEForecaster(96, horizon, variables, features,
        d_model=128, primary_idx=-1, dropout=.05).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=.5, patience=3, min_lr=1e-6)
    print('START loss={} clipping={} T={} | {} | stable_max_loops={} parameters={}'.format(
        objective, clipping, horizon, datetime.now().isoformat(),
        model.stable_block.max_loops, sum(p.numel() for p in model.parameters())), flush=True)
    best, best_state, best_epoch, stale = float('inf'), None, 0, 0
    total_steps = total_over = 0
    total_norm = total_max = 0.
    nonfinite_loss = nonfinite_grad = nonfinite_validation = 0
    failed = False
    started = time.perf_counter()
    for epoch in range(1, 101):
        epoch_start = time.perf_counter()
        model.train()
        epoch_loss = epoch_norm = epoch_max = 0.
        epoch_steps = epoch_over = 0
        for x, history, future, target in train:
            x, history, future, target = [item.cuda() for item in (x, history, future, target)]
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x, history, future)
            if objective == 'ORHL':
                absolute = (prediction - target).abs()
                loss = torch.where(absolute <= 1., .5 * absolute.square() + 5. * absolute,
                                   6. * absolute - .5).mean()
            elif objective == 'MSE':
                loss = F.mse_loss(prediction, target)
            else:
                loss = F.l1_loss(prediction, target)
            loss_value = loss.item()
            if not math.isfinite(loss_value):
                nonfinite_loss += 1
                failed = True
                print('NONFINITE_LOSS epoch={} batch={}; configuration stopped.'.format(epoch, epoch_steps + 1), flush=True)
                break
            loss.backward()
            # Observe the same global L2 gradient norm in both arms, before clipping.
            norm = torch.norm(torch.stack([p.grad.detach().norm(2)
                for p in model.parameters() if p.grad is not None]), 2).item()
            if not math.isfinite(norm):
                nonfinite_grad += 1
                failed = True
                print('NONFINITE_GRAD epoch={} batch={}; configuration stopped.'.format(epoch, epoch_steps + 1), flush=True)
                break
            if clipping == 'norm5':
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.)
            optimizer.step()
            epoch_loss += loss_value
            epoch_norm += norm
            epoch_max = max(epoch_max, norm)
            epoch_steps += 1
            epoch_over += int(norm > 5.)
        total_steps += epoch_steps
        total_norm += epoch_norm
        total_max = max(total_max, epoch_max)
        total_over += epoch_over
        if failed:
            break
        val_mse, val_mae = evaluate(model, validation)
        if not (math.isfinite(val_mse) and math.isfinite(val_mae)):
            nonfinite_validation += 1
            failed = True
            print('NONFINITE_VALIDATION epoch={}; configuration stopped.'.format(epoch), flush=True)
            break
        scheduler.step(val_mse)
        if val_mse < best:
            best, best_epoch, stale = val_mse, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        print('loss={} clipping={} T={} Epoch={:03d} train_loss={:.6f} val_MSE={:.9f} val_MAE={:.9f} preclip_grad_mean={:.6f} preclip_grad_max={:.6f} grad_gt5_fraction={:.6f} steps={} lr={:.2e} ES={}/12 seconds={:.2f}'.format(
            objective, clipping, horizon, epoch, epoch_loss / epoch_steps, val_mse, val_mae,
            epoch_norm / epoch_steps, epoch_max, epoch_over / epoch_steps, epoch_steps,
            optimizer.param_groups[0]['lr'], stale, time.perf_counter() - epoch_start), flush=True)
        if stale >= 12:
            print('Early stopping.', flush=True)
            break
    if failed or best_state is None:
        mse = mae = float('nan')
    else:
        model.load_state_dict(best_state)
        mse, mae = evaluate(model, test)
    result = dict(mse=mse, mae=mae, steps=total_steps, norm_sum=total_norm,
                  norm_max=total_max, over=total_over, nonfinite_loss=nonfinite_loss,
                  nonfinite_grad=nonfinite_grad, nonfinite_validation=nonfinite_validation,
                  failed=failed)
    print('RESULT loss={} clipping={} T={} | MSE={:.9f} MAE={:.9f} status={} best_epoch={} epochs={} seconds={:.2f}'.format(
        objective, clipping, horizon, mse, mae, 'FAILED' if failed else 'OK',
        best_epoch, epoch, time.perf_counter() - started), flush=True)
    print('STABILITY loss={} clipping={} T={} | steps={} preclip_grad_mean={:.6f} preclip_grad_max={:.6f} grad_gt5_fraction={:.6f} nonfinite_loss={} nonfinite_grad={} nonfinite_validation={}'.format(
        objective, clipping, horizon, total_steps, total_norm / max(total_steps, 1),
        total_max, total_over / max(total_steps, 1), nonfinite_loss, nonfinite_grad, nonfinite_validation), flush=True)
    del model, optimizer, scheduler, best_state, prediction, loss
    torch.cuda.empty_cache()
    return result


def main():
    sys.dont_write_bytecode = True
    torch.set_num_threads(2)
    spec = importlib.util.spec_from_file_location('norm5_weather', CODE / 'weather.py')
    source = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(source)
    print('Weather loss x clipping | Started: ' + datetime.now().isoformat(), flush=True)
    print('GPU: ' + torch.cuda.get_device_name(0), flush=True)
    print('24 fresh runs, no historical results reused. Seed=2026 L=96 T=96,192,336,720.', flush=True)
    print('D=128 batch=256 dropout=.05 stable_max_loops=2; original RK4 and all model branches retained.', flush=True)
    print('No per-loss rescaling/tuning.', flush=True)
    print('AdamW lr=5e-4 wd=1e-4; ReduceLROnPlateau factor=.5 patience=3 min_lr=1e-6; max_epochs=100 early_stop=12.', flush=True)
    print('All arms reset the same seed and independent loader generators; same initialization/order per horizon.', flush=True)
    print('Best validation MSE selects weights for all objectives. Metrics: all-channel, all test windows, train-only standardization.', flush=True)
    print('No clipping arm only observes gradient norms. norm5 arm clips global L2 parameter gradients after backward and before optimizer.step.', flush=True)
    print('Gradients are logged BEFORE clipping; gt5 is actual clipping frequency only in norm5 arm. Loss scales differ, so gradient magnitudes are not scale-independent evidence.', flush=True)
    print('Stop a configuration at its first non-finite training loss/gradient/validation metric; mark FAILED, do not omit that horizon from Avg.', flush=True)
    print('Avg MSE/MAE: unweighted arithmetic mean across four horizons; gradient summaries pool finite attempted training steps. Single-seed descriptive results.', flush=True)
    final = []
    completed = 0
    for objective in ['ORHL', 'MSE', 'MAE']:
        for clipping in ['none', 'norm5']:
            results = []
            for horizon in HORIZONS:
                results.append(run(source, objective, clipping, horizon))
                completed += 1
                print('Completed {}/24'.format(completed), flush=True)
            mse = float(np.mean([r['mse'] for r in results]))
            mae = float(np.mean([r['mae'] for r in results]))
            steps = sum(r['steps'] for r in results)
            print('AVG_STABILITY loss={} clipping={} | preclip_grad_mean={:.6f} preclip_grad_max={:.6f} grad_gt5_fraction={:.6f} failed_horizons={}'.format(
                objective, clipping, sum(r['norm_sum'] for r in results)/max(steps, 1),
                max(r['norm_max'] for r in results), sum(r['over'] for r in results)/max(steps, 1),
                sum(r['failed'] for r in results)), flush=True)
            final.append((objective, clipping, mse, mae))
            print('AVG loss={} clipping={} | MSE={:.9f} MAE={:.9f}'.format(objective, clipping, mse, mae), flush=True)
    print('FINAL WEATHER AVG RESULTS', flush=True)
    for objective, clipping, mse, mae in final:
        print('loss={} clipping={} | AVG MSE={:.9f} MAE={:.9f}'.format(objective, clipping, mse, mae), flush=True)
    print('Finished: ' + datetime.now().isoformat(), flush=True)


if __name__ == '__main__':
    with (HERE / 'weather_loss_clipping_run.log').open('x', encoding='utf-8', buffering=1) as output:
        with redirect_stdout(output), redirect_stderr(output):
            try:
                main()
            except BaseException:
                traceback.print_exc()
                raise
