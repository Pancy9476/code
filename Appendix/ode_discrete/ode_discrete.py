"""Weather L96/T96: RK4 versus gated recurrent cross-attention."""
import importlib.util
import inspect
import textwrap
import io
import sys
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
CODE = HERE.parent.parent
DATA = Path(r"D:\all_datasets\weather\weather.csv")
SEEDS = [2026, 2027, 2028]
MODES = ["RK4", "RecurrentAttention"]
LENGTH, HORIZON, WIDTH, BATCH_SIZE = 96, 96, 128, 256
MAX_EPOCHS, PATIENCE = 100, 12


def timestamp():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


class RecurrentAttention(torch.nn.Module):
    def __init__(self, width, steps=16):
        super().__init__()
        self.steps = steps
        self.q_proj = torch.nn.Linear(width, width, bias=False)
        self.proposal = torch.nn.Linear(2 * width, width)
        self.gate = torch.nn.Linear(2 * width, width)
        torch.nn.init.constant_(self.gate.bias, -2.)

    def forward(self, state, keys, values):
        for _ in range(self.steps):
            scores = self.q_proj(state) @ keys.transpose(-1, -2)
            context = torch.softmax(scores / state.shape[-1] ** .5, dim=-1) @ values
            joined = torch.cat([state, context], dim=-1)
            candidate = torch.tanh(self.proposal(joined))
            gate = torch.sigmoid(self.gate(joined))
            state = (1. - gate) * state + gate * candidate
        return state


def install_recurrent(model, source, seed):
    # Replace the entire propagation section, leaving every other forward branch intact.
    forward_source = textwrap.dedent(inspect.getsource(source.CTIAODEForecaster.forward))
    a = forward_source.index('    self.ode_func.update_exogenous(')
    b = forward_source.index('    H_T = ode_out[-1]', a) + len('    H_T = ode_out[-1]')
    forward_source = forward_source[:a] + '    H_T = self.recurrent_attention(H, K_exo, V_exo)' + forward_source[b:]
    namespace = dict(source.__dict__)
    exec(compile(forward_source, '<recurrent_attention_forward>', 'exec'), namespace)
    # Isolate new-block initialization from the common training/dropout RNG stream.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed + 40000)
        block = RecurrentAttention(WIDTH)
    del model.ode_func
    model.recurrent_attention = block
    model.forward = namespace['forward'].__get__(model, type(model))


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    squared_sum = absolute_sum = 0.0
    elements = 0
    for x, history_time, future_time, target in loader:
        prediction = model(x.cuda(), history_time.cuda(), future_time.cuda())
        residual = (prediction.cpu() - target).numpy().astype(np.float64)
        squared_sum += np.square(residual).sum()
        absolute_sum += np.abs(residual).sum()
        elements += target.numel()
    return squared_sum / elements, absolute_sum / elements


@torch.no_grad()
def count_field_calls(model, sample):
    model.eval()
    calls = [0]
    def count(*unused):
        calls[0] += 1
    target = model.recurrent_attention.proposal if hasattr(model, "recurrent_attention") else model.ode_func
    hook = target.register_forward_hook(count)
    try:
        model(*(item.cuda() for item in sample))
        torch.cuda.synchronize()
    finally:
        hook.remove()
    return calls[0]


@torch.no_grad()
def forward_cost(model, sample, batch_size):
    model.eval()
    inputs = [item[:batch_size].cuda() for item in sample]
    for _ in range(10):
        model(*inputs)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for _ in range(30):
        model(*inputs)
    torch.cuda.synchronize()
    milliseconds = (time.perf_counter() - started) * 1000 / 30
    memory_mb = torch.cuda.max_memory_allocated() / 1024**2
    return milliseconds, memory_mb


def train_one(mode, seed, initial_state=None):
    spec = importlib.util.spec_from_file_location("weather_{}_{}".format(mode, seed), CODE / "weather.py")
    source = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(source)
    source.set_seed(seed)
    with redirect_stdout(io.StringIO()):
        train, validation, test, variables, time_features = source.load_and_split_data(
            str(DATA), seq_len=LENGTH, pred_len=HORIZON, batch_size=BATCH_SIZE, num_workers=0)
    train.sampler.generator = torch.Generator().manual_seed(seed)
    train.generator = torch.Generator().manual_seed(seed + 10000)
    validation.generator = torch.Generator().manual_seed(seed + 20000)
    test.generator = torch.Generator().manual_seed(seed + 30000)
    model = source.CTIAODEForecaster(
        LENGTH, HORIZON, variables, time_features, d_model=WIDTH,
        primary_idx=-1, dropout=.05)
    if initial_state is None:
        initial_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    else:
        # Load the exact same initial weights as this seed's RK4 model.
        model.load_state_dict(initial_state)
    if mode == "RecurrentAttention":
        install_recurrent(model, source, seed)
    model.cuda()
    sample_batch = next(iter(validation))
    sample = sample_batch[:3]
    parameters = sum(p.numel() for p in model.parameters())
    calls = count_field_calls(model, sample)
    expected_calls = 16
    if calls != expected_calls:
        raise RuntimeError("Expected {} field calls; got {}".format(expected_calls, calls))
    print("START mode={} seed={} | {}".format(mode, seed, timestamp()), flush=True)
    print("parameters={} propagation_evaluations_per_forward={} stable_max_loops={} test_windows={}".format(
        parameters, calls, model.stable_block.max_loops, len(test.dataset)), flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=.5, patience=3, min_lr=1e-6)
    best_mse, best_epoch, stale, best_state = float("inf"), 0, 0, None
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()

    for epoch in range(1, MAX_EPOCHS + 1):
        epoch_start = time.perf_counter()
        model.train()
        train_loss = 0.0
        for x, history_time, future_time, target in train:
            x, history_time, future_time, target = [
                item.cuda(non_blocking=True) for item in (x, history_time, future_time, target)]
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x, history_time, future_time)
            absolute = (prediction - target).abs()
            loss = torch.where(absolute <= 1., .5 * absolute.square() + 5. * absolute,
                               6. * absolute - .5).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.)
            optimizer.step()
            train_loss += loss.item()
        val_mse, val_mae = evaluate(model, validation)
        scheduler.step(val_mse)
        improved = val_mse < best_mse
        if improved:
            best_mse, best_epoch, stale = val_mse, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        print("mode={} seed={} | Epoch {:03d} | val_MSE={:.9f} val_MAE={:.9f} | lr={:.2e} | ES={}/{} | seconds={:.2f} {}".format(
            mode, seed, epoch, val_mse, val_mae,
            optimizer.param_groups[0]["lr"], stale, PATIENCE,
            time.perf_counter() - epoch_start, "BEST" if improved else ""), flush=True)
        if stale >= PATIENCE:
            print("Early stopping.", flush=True)
            break

    training_seconds = time.perf_counter() - started
    train_peak_mb = torch.cuda.max_memory_allocated() / 1024**2
    model.load_state_dict(best_state)
    mse, mae = evaluate(model, test)
    # Release training-only tensors and optimizer states before forward-memory measurements.
    model.zero_grad(set_to_none=True)
    del optimizer, scheduler, prediction, loss, absolute, x, history_time, future_time, target
    torch.cuda.empty_cache()
    latency1, memory1 = forward_cost(model, sample, 1)
    latency256, memory256 = forward_cost(model, sample, BATCH_SIZE)
    result = dict(mode=mode, seed=seed, mse=mse, mae=mae, parameters=parameters,
                  field_calls=calls, train_val_seconds=training_seconds,
                  train_peak_mb=train_peak_mb, batch1_ms=latency1, batch256_ms=latency256,
                  forward_peak_mb_batch1=memory1, forward_peak_mb_batch256=memory256)
    print("RESULT mode={} seed={} | MSE={:.9f} MAE={:.9f} | best_epoch={} epochs={} | train_val_seconds={:.2f} train_peak_MB={:.2f}".format(
        mode, seed, mse, mae, best_epoch, epoch, training_seconds, train_peak_mb), flush=True)
    print("FORWARD mode={} seed={} | batch1_ms={:.6f} batch256_ms={:.6f} | peak_MB_batch1={:.2f} peak_MB_batch256={:.2f}".format(
        mode, seed, latency1, latency256, memory1, memory256), flush=True)
    del model, best_state, sample, sample_batch
    torch.cuda.empty_cache()
    return result, initial_state


def main():
    sys.dont_write_bytecode = True
    torch.set_num_threads(2)
    print('Weather L96 T96 | RK4 vs RecurrentAttention | Started: ' + timestamp(), flush=True)
    print('GPU: ' + torch.cuda.get_device_name(0), flush=True)
    print('Paired seeds=2026,2027,2028; 6 fresh training runs; no old test results reused.', flush=True)
    print('RecurrentAttention: 16 shared gated cross-attention steps; candidate=tanh(Wc[H,C]+bc), gate=sigmoid(Wg[H,C]+bg), H_next=(1-gate)*H+gate*candidate.', flush=True)
    print('Context K/V and all other model branches retained; original ode_func removed. No time grid, solver or vector-field call in recurrent forward.', flush=True)
    print('RK4: 16 field evaluations; recurrent: 16 attention/update steps. Counts comparable, not identical operations/FLOPs; runtime measured separately.', flush=True)
    print('Recurrent gate bias initialized to -2; remaining new weights use standard Linear initialization, seed+40000.', flush=True)
    print('Common branches explicitly share initial weights per seed; new recurrent block has its own parameters.', flush=True)
    print('D=128 L=96 T=96 batch=256 dropout=.05 stable_max_loops=2 threads=2 FP32', flush=True)
    print('AdamW lr=5e-4 wd=1e-4 clip_norm=5 max_epochs=100 patience=12; scheduler factor=.5 patience=3 min_lr=1e-6.', flush=True)
    print('Best validation MSE selects weights; all-channel standardized test MSE/MAE.', flush=True)
    results = []
    for seed in SEEDS:
        initial_state = None
        pair = []
        for mode in ['RK4', 'RecurrentAttention']:
            result, initial_state = train_one(mode, seed, initial_state)
            results.append(result)
            pair.append(result)
        print('PARAMETER_COMPARISON RK4={} RecurrentAttention={} difference={:+d}'.format(pair[0]['parameters'], pair[1]['parameters'], pair[1]['parameters']-pair[0]['parameters']), flush=True)
        print('PAIRED seed={} | recurrent_minus_rk4_MSE={:+.9f} MAE={:+.9f}'.format(seed, pair[1]['mse']-pair[0]['mse'], pair[1]['mae']-pair[0]['mae']), flush=True)
        del initial_state
    print('FINAL RESULTS: mean and sample standard deviation across 3 seeds, T96', flush=True)
    for mode in ['RK4', 'RecurrentAttention']:
        for metric in ['mse', 'mae', 'batch1_ms', 'batch256_ms', 'train_peak_mb']:
            values = np.array([r[metric] for r in results if r['mode'] == mode])
            print('{} {} | mean={:.9f} std={:.9f}'.format(mode, metric, values.mean(), values.std(ddof=1)), flush=True)
    print('Finished: ' + timestamp(), flush=True)


if __name__ == '__main__':
    with (HERE / 'ode_discrete.log').open('x', encoding='utf-8', buffering=1) as output:
        with redirect_stdout(output), redirect_stderr(output):
            try:
                main()
            except BaseException:
                traceback.print_exc()
                raise
