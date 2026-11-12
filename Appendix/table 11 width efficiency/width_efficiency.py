"""Weather L96 T96: CTIA-ODE/TimeXer at widths 128 and 512 on one GPU."""
import gc
import importlib.util
import io
import re
import sys
import time
import traceback
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
CODE = HERE.parent.parent
LIBRARY = CODE.parent / 'work2/third_party/Time-Series-Library'
sys.path.insert(0, str(LIBRARY))
from models import TimeXer

DATA = Path(r'D:\all_datasets\weather\weather.csv')
SEED, BATCH = 2026, 256


def load_source():
    spec = importlib.util.spec_from_file_location('width_weather', CODE / 'weather.py')
    source = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(source)
    return source


def make_model(source, name, width, channels, features):
    if name == 'CTIA-ODE':
        model = source.CTIAODEForecaster(96, 96, channels, features,
            d_model=width, primary_idx=-1, dropout=.05)
        model.stable_block.max_loops = 2
        return model
    config = SimpleNamespace(task_name='long_term_forecast', features='M',
        seq_len=96, pred_len=96, enc_in=channels, use_norm=True, patch_len=16,
        d_model=width, d_ff=4*width, e_layers=1, n_heads=8, factor=3,
        embed='timeF', freq='t', dropout=.1, activation='gelu')
    return TimeXer.Model(config)


def forward(model, name, batch):
    x, history, future = batch[:3]
    if name == 'CTIA-ODE':
        return model(x, history, future)
    # TimeXer encoder-only multivariate path uses history/calendar, not decoder values.
    return model(x.transpose(1, 2).contiguous(), history.transpose(1, 2).contiguous(), None, None).transpose(1, 2)


@torch.no_grad()
def evaluate(model, name, loader):
    model.eval()
    squared = absolute = 0.
    count = 0
    for batch in loader:
        prediction = forward(model, name, [v.cuda() for v in batch[:3]])
        error = (prediction.cpu()-batch[3]).numpy().astype(np.float64)
        squared += np.square(error).sum()
        absolute += np.abs(error).sum()
        count += error.size
    return squared/count, absolute/count


def objective(prediction, target):
    absolute = torch.abs(prediction-target)
    return torch.where(absolute <= 1., .5*absolute**2+5.*absolute, 6.*absolute-.5).mean()


def clean_cuda():
    gc.collect()
    torch.cuda.empty_cache()


@torch.no_grad()
def benchmark(source, name, width, channels, features, weights, sample, batch_size):
    model = make_model(source, name, width, channels, features).cuda().eval()
    model.load_state_dict(weights)
    inputs = [v[:batch_size].cuda() for v in sample]
    for _ in range(10):
        forward(model, name, inputs)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for _ in range(30):
        forward(model, name, inputs)
    torch.cuda.synchronize()
    latency = (time.perf_counter()-started)*1000/30
    memory = torch.cuda.max_memory_allocated()/1024**2
    del model, inputs
    clean_cuda()
    return latency, memory


def probe(source, channels, features, sample):
    for name in ['TimeXer', 'CTIA-ODE']:
        source.set_seed(SEED)
        model = make_model(source, name, 512, channels, features).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
        batch = [v.cuda() for v in sample]
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            prediction = forward(model, name, batch[:3])
            loss = objective(prediction, batch[3])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            forward(model, name, batch[:3])
        torch.cuda.synchronize()
        print('PROBE {} width=512 batch={} forward/backward/AdamW/eval OK peak_allocated_MB={:.2f} seconds={:.2f}'.format(
            name, BATCH, torch.cuda.max_memory_allocated()/1024**2, time.perf_counter()-started), flush=True)
        del model, optimizer, batch, prediction, loss
        clean_cuda()


def run(source, name, width):
    source.set_seed(SEED)
    with redirect_stdout(io.StringIO()):
        train, val, test, channels, features = source.load_and_split_data(
            str(DATA), seq_len=96, pred_len=96, batch_size=BATCH, num_workers=0)
    train.sampler.generator = torch.Generator().manual_seed(SEED)
    for i, loader in enumerate([train, val, test]):
        loader.generator = torch.Generator().manual_seed(SEED+10000*(i+1))
    model = make_model(source, name, width, channels, features).cuda()
    parameters = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=.5, patience=3, min_lr=1e-6)
    best, weights, stale, best_epoch = float('inf'), None, 0, 0
    print('START {} width={} parameters={} | {}'.format(name, width, parameters, datetime.now().isoformat()), flush=True)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for epoch in range(1, 101):
        epoch_start = time.perf_counter()
        model.train()
        for batch in train:
            inputs = [v.cuda() for v in batch[:3]]
            target = batch[3].cuda()
            optimizer.zero_grad(set_to_none=True)
            prediction = forward(model, name, inputs)
            loss = objective(prediction, target)
            if not torch.isfinite(loss).item():
                raise RuntimeError('Non-finite training objective')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            if not torch.isfinite(norm).item():
                raise RuntimeError('Non-finite parameter gradients')
            optimizer.step()
        mse, mae = evaluate(model, name, val)
        if not np.isfinite(mse):
            raise RuntimeError('Non-finite validation MSE')
        scheduler.step(mse)
        if mse < best:
            best, stale, best_epoch = mse, 0, epoch
            weights = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            stale += 1
        print('{} width={} Epoch={:03d} val_MSE={:.9f} val_MAE={:.9f} ES={}/12 seconds={:.2f}'.format(
            name, width, epoch, mse, mae, stale, time.perf_counter()-epoch_start), flush=True)
        if stale >= 12:
            break
    training_seconds = time.perf_counter()-started
    training_memory = torch.cuda.max_memory_allocated()/1024**2
    model.load_state_dict(weights)
    mse, mae = evaluate(model, name, test)
    sample = next(iter(val))[:3]
    del model, optimizer, scheduler, inputs, target, prediction, loss, norm, batch
    clean_cuda()
    latency1, memory1 = benchmark(source, name, width, channels, features, weights, sample, 1)
    latency256, memory256 = benchmark(source, name, width, channels, features, weights, sample, BATCH)
    print('TRAINING {} width={} best_epoch={} epochs={} seconds={:.2f} peak_allocated_MB={:.2f}'.format(
        name, width, best_epoch, epoch, training_seconds, training_memory), flush=True)
    print('RESULT {} width={} | MSE={:.9f} MAE={:.9f} parameters={} batch1_ms={:.6f} batch1_peak_MB={:.3f} batch256_ms={:.6f} batch256_peak_MB={:.3f}'.format(
        name, width, mse, mae, parameters, latency1, memory1, latency256, memory256), flush=True)
    del weights
    return mse, mae, parameters, latency1, memory1, latency256, memory256


def main():
    torch.set_num_threads(2)
    source = load_source()
    print('Width/resource comparison | '+datetime.now().isoformat(), flush=True)
    print('GPU: '+torch.cuda.get_device_name(0)+' | torch='+torch.__version__, flush=True)
    print('TimeXer implementation: '+str(LIBRARY/'models/TimeXer.py'), flush=True)
    print('Weather L96 T96 seed2026, all-channel multivariate forecasting, train-only standardization; four fresh runs.', flush=True)
    print('Widths128/512. CTIA uses source branches and stable_max_loops2; TimeXer uses multivariate features=M, patch16, layers1, heads8, d_ff=4*d_model, dropout.1.', flush=True)
    print('TimeXer width/FF ratio is fixed in this new sweep; these configs are not claimed to reproduce its paper figure or every repository shell-script setting.', flush=True)
    print('Both use the same eight source calendar features; each architecture retains its own use of those features.', flush=True)
    print('Common protocol: batch256 FP32 AdamW lr5e-4 wd1e-4 ORHL=Huber(delta1)+5*MAE norm5 max_epochs100 early_stop12; plateau factor.5 patience3 min_lr1e-6.', flush=True)
    print('Same seed/reset and sampler scheme; no tuning; best validation MSE checkpoint; test not used for selection.', flush=True)
    print('Inference: batch1/batch256, fresh best-weight model after releasing optimizer/training tensors; eval/no_grad; inputs resident on GPU; 10 warmups, 30 synchronized repeats.', flush=True)
    print('Latency is mean ms/batch, excludes transfer/metrics. GPU memory is peak PyTorch allocated MiB, includes model/inputs/cache/activations, excludes reserved memory and external GPU allocations.', flush=True)
    print('Training peak allocated memory reported separately. No literature runtime/memory values mixed into local comparison. Single seed, two-width samples, not a complete Pareto frontier.', flush=True)
    with redirect_stdout(io.StringIO()):
        _, val, _, channels, features = source.load_and_split_data(str(DATA), seq_len=96, pred_len=96, batch_size=BATCH, num_workers=0)
    sample = next(iter(val))
    probe(source, channels, features, sample)
    del val, sample
    results = []
    for name, width in [('TimeXer',128), ('TimeXer',512), ('CTIA-ODE',128), ('CTIA-ODE',512)]:
        result = run(source, name, width)
        results.append((name,width,result))
    print('FINAL SUMMARY', flush=True)
    for name,width,result in results:
        print('{} width={} | MSE={:.9f} MAE={:.9f} parameters={} batch1_ms={:.6f} batch1_peak_MB={:.3f} batch256_ms={:.6f} batch256_peak_MB={:.3f}'.format(name,width,*result),flush=True)
    print('Finished: '+datetime.now().isoformat(),flush=True)


if __name__=='__main__':
    sys.dont_write_bytecode=True
    with (HERE/'width_efficiency.log').open('x',encoding='utf-8',buffering=1) as output:
        with redirect_stdout(output),redirect_stderr(output):
            try:
                main()
            except BaseException:
                traceback.print_exc()
                raise
