import importlib.util
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch


HERE = Path(__file__).resolve().parent
CODE = HERE.parent
DATA = Path(r"D:\all_datasets\ETT-small\ETTm2.csv")
WEIGHTS = CODE / "weights/ETTm2/best_model_T96.pth"
ORIGINAL_MSE = 0.174985  
SEED = 2026


def main():
    sys.dont_write_bytecode = True
    torch.set_num_threads(4)
    spec = importlib.util.spec_from_file_location("ettm2", CODE / "ettm2.py")
    source = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(source)
    source.set_seed(SEED)
    device = torch.device("cuda")

    with (HERE / "table4_run.log").open("w", encoding="utf-8") as output:
        def log(message):
            print(message, flush=True)
            output.write(message + "\n")
            output.flush()

        log("Table 4: CTIA-ODE, ETTm2, existing-checkpoint evaluation")
        log("Started: " + datetime.now(timezone(timedelta(hours=8))).isoformat())
        log("Source: ettm2.py | Weights: weights/ETTm2/best_model_T96.pth")
        log("L=96 | T=96 | d_model=16 | batch_size=120 | seed=2026")
        log("Metrics: all 7 channels, train-standardized values, all test windows")
        log("Mask: independent probability 0.30, exogenous input only, standardized zero")
        log("Shift+2: exogenous x(t-2), earlier real history, no wrap or padding")
        log("OT, targets and calendar features are unchanged.")
        log("Historical Original MSE reference: {:.6f} (not rerun)".format(ORIGINAL_MSE))
        _, _, loader, variables, time_features = source.load_and_split_data(
            str(DATA), seq_len=96, pred_len=96, batch_size=120, num_workers=0)
        values = pd.read_csv(DATA).drop(columns=["date"]).to_numpy(dtype=np.float32)
        scaler = source.StandardScaler()
        scaler.fit(values[:34560])
        scaled = scaler.transform(values)
        model = source.CTIAODEForecaster(
            96, 96, variables, time_features, d_model=16, primary_idx=-1, dropout=0.1)
        model.load_state_dict(torch.load(str(WEIGHTS), map_location="cpu"))
        model.to(device).eval()
        log("Test windows: {}".format(len(loader.dataset)))

        for condition in ["Mask-30%", "Shift+2"]:
            rng = np.random.RandomState(SEED)
            squared_sum = absolute_sum = 0.0
            elements = samples = 0
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.no_grad():
                for batch, (x, history_time, future_time, target) in enumerate(loader, 1):
                    perturbed = x.clone()
                    if condition == "Mask-30%":
                        mask = rng.random_sample((len(x), 6, 96)) < 0.30
                        perturbed[:, :6, :].masked_fill_(torch.from_numpy(mask), 0.0)
                    else:
                        starts = range(46080 - 96 + samples, 46080 - 96 + samples + len(x))
                        delayed = np.stack([scaled[s - 2:s + 94, :6].T for s in starts])
                        perturbed[:, :6, :] = torch.from_numpy(delayed)
                    pred = model(perturbed.to(device), history_time.to(device), future_time.to(device))
                    residual = (pred.cpu() - target).numpy().astype(np.float64)
                    squared_sum += np.square(residual).sum()
                    absolute_sum += np.abs(residual).sum()
                    elements += target.numel()
                    samples += len(x)
                    if batch % 20 == 0 or batch == len(loader):
                        log("{} | batch {}/{} | windows {}".format(condition, batch, len(loader), samples))
            torch.cuda.synchronize()
            mse, mae = squared_sum / elements, absolute_sum / elements
            delta = (mse / ORIGINAL_MSE - 1) * 100
            log("{} | MSE={:.9f} | MAE={:.9f} | Delta MSE={:+.4f}% | seconds={:.2f}".format(
                condition, mse, mae, delta, time.perf_counter() - started))
        log("Finished: " + datetime.now(timezone(timedelta(hours=8))).isoformat())


if __name__ == "__main__":
    main()
