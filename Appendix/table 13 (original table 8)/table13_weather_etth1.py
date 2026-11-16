"""表8：CTIA-ODE，Weather/ETTh1，各输入长度取四个预测长度的Avg。"""
import importlib.util
import re
import sys
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


HERE = Path(__file__).resolve().parent
CODE = HERE
DATA_ROOT = HERE
LOOKBACKS = [96, 192, 336, 720]
HORIZONS = [96, 192, 336, 720]
SETTINGS = [
    ("Weather", "weather.py", "weather.csv", 128, 256, .05, 5e-4, 1e-4, 100, 1e-6),
    ("ETTh1", "etth1.py", "ETTh1.csv", 16, 256, .1, 2e-5, 1e-3, 1000, 1e-8),
]
SEED, PATIENCE = 2026, 12


def timestamp():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


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


def train_one(source, config, lookback, horizon):
    dataset, _, datafile, width, batch_size, dropout, lr, decay, max_epochs, min_lr = config
    source.set_seed(SEED)
    train, validation, test, variables, time_features = source.load_and_split_data(
        str(DATA_ROOT / datafile), seq_len=lookback, pred_len=horizon,
        batch_size=batch_size, num_workers=0)
    train.sampler.generator = torch.Generator().manual_seed(SEED)
    train.generator = torch.Generator().manual_seed(SEED + 10000)
    validation.generator = torch.Generator().manual_seed(SEED + 20000)
    test.generator = torch.Generator().manual_seed(SEED + 30000)
    model = source.CTIAODEForecaster(
        lookback, horizon, variables, time_features, d_model=width,
        primary_idx=-1, dropout=dropout).cuda()
    if dataset == "ETTh1":
        model.stable_block.max_loops = 2
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=.5, patience=3, min_lr=min_lr)
    best_mse, best_epoch, stale, best_state = float("inf"), 0, 0, None
    started = time.perf_counter()
    print("START {} L={} T={} | {}".format(dataset, lookback, horizon, timestamp()), flush=True)
    print("d_model={} batch_size={} dropout={} lr={} weight_decay={} max_epochs={} min_lr={} stable_max_loops={}".format(
        width, batch_size, dropout, lr, decay, max_epochs, min_lr, model.stable_block.max_loops), flush=True)
    print("train_windows={} val_windows={} test_windows={}".format(
        len(train.dataset), len(validation.dataset), len(test.dataset)), flush=True)

    for epoch in range(1, max_epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        train_loss = 0.0
        for x, history_time, future_time, target in train:
            x, history_time, future_time, target = [
                item.cuda(non_blocking=True) for item in (x, history_time, future_time, target)]
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x, history_time, future_time)
            if dataset == "Weather":
                absolute = (prediction - target).abs()
                loss = torch.where(absolute <= 1., .5 * absolute.square() + 5. * absolute,
                                   6. * absolute - .5).mean()
            else:
                loss = F.huber_loss(prediction, target, delta=1.) + 5. * F.l1_loss(prediction, target)
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
        print("{} L={} T={} | Epoch {:03d} | train_loss={:.6f} | val_MSE={:.9f} val_MAE={:.9f} | lr={:.2e} | ES={}/{} | seconds={:.2f} {}".format(
            dataset, lookback, horizon, epoch, train_loss / len(train), val_mse, val_mae,
            optimizer.param_groups[0]["lr"], stale, PATIENCE,
            time.perf_counter() - epoch_start, "BEST" if improved else ""), flush=True)
        if stale >= PATIENCE:
            print("Early stopping.", flush=True)
            break
    model.load_state_dict(best_state)
    mse, mae = evaluate(model, test)
    print("RESULT {} L={} T={} | MSE={:.9f} MAE={:.9f} | best_epoch={} epochs={} | seconds={:.2f}".format(
        dataset, lookback, horizon, mse, mae, best_epoch, epoch, time.perf_counter() - started), flush=True)
    del model, optimizer, scheduler, best_state
    torch.cuda.empty_cache()
    return mse, mae


def main(completed_weather):
    sys.dont_write_bytecode = True
    torch.set_num_threads(2)
    print("Table 8: CTIA-ODE look-back length experiment", flush=True)
    print("Started: " + timestamp(), flush=True)
    print("Datasets: Weather, ETTh1 | L=96,192,336,720 | T=96,192,336,720", flush=True)
    print("32 training configurations; original model definitions from weather.py and etth1.py", flush=True)
    print("ETTh1 setting: batch_size=256, stable_max_loops=2 (requested update)", flush=True)
    print("Completed Weather results retained from earlier log entries: {}".format(len(completed_weather)), flush=True)
    print("Avg: arithmetic mean of the four test MSEs and four test MAEs for each dataset and look-back length", flush=True)
    print("Seed=2026 per training run | CPU threads=2 (Table 7 runs concurrently)", flush=True)
    print("GPU: " + torch.cuda.get_device_name(0), flush=True)
    print("Loss=Huber(delta=1)+5*L1 | clip_norm=5 | AdamW | early_stop_patience=12", flush=True)
    print("ReduceLROnPlateau factor=0.5 patience=3; dataset-specific settings logged below", flush=True)
    print("Metrics: all variables, all test windows, train-only standardization", flush=True)
    print("Each look-back independently trained. Best validation MSE selects in-memory weights; test does not select weights.", flush=True)
    print("Only completed Weather results from this same experiment are reused when resuming; ETTh1 is trained with the updated settings.", flush=True)
    print("Baseline models Autoformer and DLinear are not run in this script.", flush=True)
    all_results = []
    completed = 0
    started = time.perf_counter()
    for config in SETTINGS:
        dataset, filename = config[:2]
        spec = importlib.util.spec_from_file_location("table8_" + dataset, CODE / filename)
        source = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(source)
        for lookback in LOOKBACKS:
            horizon_results = []
            for horizon in HORIZONS:
                if dataset == "Weather" and (lookback, horizon) in completed_weather:
                    mse, mae = completed_weather[(lookback, horizon)]
                    print("REUSED completed Weather L={} T={} | MSE={:.9f} MAE={:.9f} | source: earlier RESULT entry in this log".format(
                        lookback, horizon, mse, mae), flush=True)
                else:
                    mse, mae = train_one(source, config, lookback, horizon)
                horizon_results.append((mse, mae))
                completed += 1
                print("Completed {}/32 training runs".format(completed), flush=True)
            avg_mse = float(np.mean([pair[0] for pair in horizon_results]))
            avg_mae = float(np.mean([pair[1] for pair in horizon_results]))
            all_results.append((dataset, lookback, avg_mse, avg_mae))
            print("AVG {} L={} | MSE={:.9f} MAE={:.9f}".format(
                dataset, lookback, avg_mse, avg_mae), flush=True)
    print("\nFINAL TABLE 8 AVG RESULTS", flush=True)
    for dataset, lookback, mse, mae in all_results:
        print("{} L={} | AVG MSE={:.9f} MAE={:.9f}".format(dataset, lookback, mse, mae), flush=True)
    print("Finished: {} | total_seconds={:.2f}".format(timestamp(), time.perf_counter() - started), flush=True)


if __name__ == "__main__":
    log_path = HERE / "table8_run.log"
    resume_weather = "--resume-weather" in sys.argv
    completed_weather = {}
    if resume_weather and log_path.is_file():
        entries = re.findall(r"^RESULT Weather L=(\d+) T=(\d+) \| MSE=([\d.eE+-]+) MAE=([\d.eE+-]+)",
                             log_path.read_text(encoding="utf-8"), re.MULTILINE)
        completed_weather = {(int(l), int(t)): (float(mse), float(mae)) for l, t, mse, mae in entries}
    with log_path.open("a" if resume_weather else "w", encoding="utf-8", buffering=1) as output:
        with redirect_stdout(output), redirect_stderr(output):
            try:
                if resume_weather:
                    print("\nRESUME: retain completed Weather runs; ETTh1 updated to loops=2, batch=256", flush=True)
                main(completed_weather)
            except BaseException:
                traceback.print_exc()
                raise
