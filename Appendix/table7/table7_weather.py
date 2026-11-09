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


HERE = Path(__file__).resolve().parent
CODE = HERE.parent.parent
DATA = Path(r"D:\all_datasets\weather\weather.csv")
TABLE8_LOG = HERE.parent / "table8/table8_run.log"
HORIZONS = [96, 192, 336, 720]
PARAMETERS = [(1.0, 5.0)] + [
    (delta, weight) for delta in [0.5, 1.0, 2.0]
    for weight in [1.0, 5.0, 10.0] if (delta, weight) != (1.0, 5.0)
]
SEED, BATCH_SIZE, WIDTH = 2026, 256, 128
MAX_EPOCHS, PATIENCE = 100, 12
LEARNING_RATE, WEIGHT_DECAY = 5e-4, 1e-4


def timestamp():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def default_results_from_table8():
    if not TABLE8_LOG.is_file():
        return {}
    text = TABLE8_LOG.read_text(encoding="utf-8")
    matches = re.findall(r"^RESULT Weather L=96 T=(\d+) \| MSE=([\d.eE+-]+) MAE=([\d.eE+-]+)", text, re.MULTILINE)
    results = {int(horizon): (float(mse), float(mae)) for horizon, mse, mae in matches}
    return results if set(results) == set(HORIZONS) else {}


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


def train_one(source, delta, weight, horizon):
    source.set_seed(SEED)
    train, validation, test, variables, time_features = source.load_and_split_data(
        str(DATA), seq_len=96, pred_len=horizon, batch_size=BATCH_SIZE, num_workers=0)
    train.sampler.generator = torch.Generator().manual_seed(SEED)
    train.generator = torch.Generator().manual_seed(SEED + 10000)
    validation.generator = torch.Generator().manual_seed(SEED + 20000)
    test.generator = torch.Generator().manual_seed(SEED + 30000)
    model = source.CTIAODEForecaster(
        96, horizon, variables, time_features, d_model=WIDTH,
        primary_idx=-1, dropout=0.05).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6)
    best_mse, best_epoch, stale, best_state = float("inf"), 0, 0, None
    started = time.perf_counter()
    print("START delta={} lambda={} T={} | {}".format(delta, weight, horizon, timestamp()), flush=True)
    print("train_windows={} val_windows={} test_windows={} stable_max_loops={}".format(
        len(train.dataset), len(validation.dataset), len(test.dataset), model.stable_block.max_loops), flush=True)
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
            loss = torch.where(absolute <= delta,
                               .5 * absolute.square() + weight * absolute,
                               (delta + weight) * absolute - .5 * delta**2).mean()
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
        print("delta={} lambda={} T={} | Epoch {:03d} | train_loss={:.6f} | val_MSE={:.9f} val_MAE={:.9f} | lr={:.2e} | ES={}/{} | seconds={:.2f} {}".format(
            delta, weight, horizon, epoch, train_loss / len(train), val_mse, val_mae,
            optimizer.param_groups[0]["lr"], stale, PATIENCE,
            time.perf_counter() - epoch_start, "BEST" if improved else ""), flush=True)
        if stale >= PATIENCE:
            print("Early stopping.", flush=True)
            break
    model.load_state_dict(best_state)
    mse, mae = evaluate(model, test)
    print("RESULT delta={} lambda={} T={} | MSE={:.9f} MAE={:.9f} | best_epoch={} epochs={} | seconds={:.2f}".format(
        delta, weight, horizon, mse, mae, best_epoch, epoch, time.perf_counter() - started), flush=True)
    del model, optimizer, scheduler, best_state
    torch.cuda.empty_cache()
    return mse, mae


def main():
    sys.dont_write_bytecode = True
    torch.set_num_threads(2)
    spec = importlib.util.spec_from_file_location("table7_weather_source", CODE / "weather.py")
    source = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(source)
    reused = default_results_from_table8()
    print("Table 7: Weather, loss hyperparameter grid, four-horizon Avg", flush=True)
    print("Started: " + timestamp(), flush=True)
    print("Source: weather.py | Dataset: weather.csv | GPU: " + torch.cuda.get_device_name(0), flush=True)
    print("L=96 | T=96,192,336,720 | d_model=128 | batch_size=256 | dropout=0.05 | stable_max_loops=2", flush=True)
    print("Loss delta=0.5,1,2 | lambda=1,5,10", flush=True)
    print("AdamW lr=0.0005 weight_decay=0.0001 | clip_norm=5 | max_epochs=100 | early_stop_patience=12", flush=True)
    print("ReduceLROnPlateau: factor=0.5 patience=3 min_lr=0.000001 | CPU threads=2", flush=True)
    print("Seed=2026 per training run, independent shuffle generators", flush=True)
    print("Metrics: all 21 channels, train-only standardization, all test windows", flush=True)
    print("Avg: arithmetic mean of the four test MSEs and four test MAEs", flush=True)
    all_results = []
    started = time.perf_counter()
    for group, (delta, weight) in enumerate(PARAMETERS, 1):
        print("\nGROUP {}/9: delta={} lambda={}".format(group, delta, weight), flush=True)
        results = []
        for horizon in HORIZONS:
            if delta == 1. and weight == 5. and reused:
                mse, mae = reused[horizon]
                print("REUSED table8/table8_run.log: Weather L=96 T={} delta=1 lambda=5 | MSE={:.9f} MAE={:.9f}".format(
                    horizon, mse, mae), flush=True)
            else:
                mse, mae = train_one(source, delta, weight, horizon)
            results.append((mse, mae))
        avg_mse = float(np.mean([pair[0] for pair in results]))
        avg_mae = float(np.mean([pair[1] for pair in results]))
        all_results.append((delta, weight, avg_mse, avg_mae))
        print("AVG delta={} lambda={} | MSE={:.9f} MAE={:.9f} | groups_completed={}/9".format(
            delta, weight, avg_mse, avg_mae, group), flush=True)
    print("\nFINAL AVG RESULTS", flush=True)
    for delta, weight, mse, mae in sorted(all_results):
        print("delta={} lambda={} | AVG MSE={:.9f} MAE={:.9f}".format(delta, weight, mse, mae), flush=True)
    print("Finished: {} | total_seconds={:.2f}".format(timestamp(), time.perf_counter() - started), flush=True)


if __name__ == "__main__":
    with (HERE / "table7_run.log").open("w", encoding="utf-8", buffering=1) as output:
        with redirect_stdout(output), redirect_stderr(output):
            try:
                main()
            except BaseException:
                traceback.print_exc()
                raise
