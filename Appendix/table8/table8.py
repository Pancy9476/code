"""Table8 CTIA-ODE ETTm1, four look-backs, four horizons, averaged metrics."""
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
CODE = HERE.parent.parent
DATA_ROOT = Path(r"D:\all_datasets")
LOOKBACKS = [96, 192, 336, 720]
HORIZONS = [96, 192, 336, 720]
SETTINGS = [
    ("ETTm1", "ettm1.py", "ETT-small/ETTm1.csv", 16, 256, .1, 2e-5, 1e-3, 1000, 1e-8),
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
    if dataset == "ETTm1":
        model.stable_block.max_loops = 2
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=.5, patience=3, min_lr=min_lr)
    best_mse, best_epoch, stale, best_state = float("inf"), 0, 0, None
    started = time.perf_counter()
    print("START {} L={} T={} | {}".format(dataset, lookback, horizon, timestamp()), flush=True)

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
    print("Table 8: CTIA-ODE ETTm1 look-back length experiment", flush=True)
    print("Started: " + timestamp(), flush=True)
    print("Dataset: ETTm1 | L=96,192,336,720 | T=96,192,336,720", flush=True)
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
                mse, mae = train_one(source, config, lookback, horizon)
                horizon_results.append((mse, mae))
                completed += 1
                print("Completed {}/16 training runs".format(completed), flush=True)
            avg_mse = float(np.mean([pair[0] for pair in horizon_results]))
            avg_mae = float(np.mean([pair[1] for pair in horizon_results]))
            all_results.append((dataset, lookback, avg_mse, avg_mae))
            print("AVG {} L={} | MSE={:.9f} MAE={:.9f}".format(
                dataset, lookback, avg_mse, avg_mae), flush=True)
    print("\nFINAL TABLE 8 ETTm1 CTIA-ODE AVG RESULTS", flush=True)
    for dataset, lookback, mse, mae in all_results:
        print("{} L={} | AVG MSE={:.9f} MAE={:.9f}".format(dataset, lookback, mse, mae), flush=True)
    print("Finished: {} | total_seconds={:.2f}".format(timestamp(), time.perf_counter() - started), flush=True)


if __name__ == "__main__":
    with (HERE / "table8_run.log").open("x", encoding="utf-8", buffering=1) as output:
        with redirect_stdout(output), redirect_stderr(output):
            try:
                main({})
            except BaseException:
                traceback.print_exc()
                raise
