"""表6：固定已有权重，测试 Euler / Midpoint / RK4，记录到 table6_run.log。"""
import importlib.util
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch
from torchdiffeq import odeint


HERE = Path(__file__).resolve().parent
CODE = HERE.parent.parent
DATA_ROOT = Path(r"D:\all_datasets")
SETTINGS = [
    ("Weather", "weather.py", "weather/weather.csv", "weather", 128, 256, 0.05),
    ("ETTm2", "ettm2.py", "ETT-small/ETTm2.csv", "ETTm2", 16, 120, 0.1),
]
SOLVERS = [("Euler", "euler", 4), ("Midpoint", "midpoint", 8), ("RK4", "rk4", 16)]


def main():
    sys.dont_write_bytecode = True
    torch.set_num_threads(4)
    device = torch.device("cuda")

    with (HERE / "table6_run.log").open("w", encoding="utf-8") as output:
        def log(message):
            print(message, flush=True)
            output.write(message + "\n")
            output.flush()

        log("Table 6: CTIA-ODE, fixed existing weights, inference solver sensitivity")
        log("Started: " + datetime.now(timezone(timedelta(hours=8))).isoformat())
        log("No retraining. One existing T720 checkpoint per dataset, reused across all solvers.")
        log("L=96 | T=720 | seed=2026 | eval mode | float32 predictions")
        log("Integration grid: [0, 0.25, 0.5, 0.75, 1], four intervals")
        log("Forward NFE: Euler=4, Midpoint=8, RK4=16")
        log("RK4: torchdiffeq rk4 implementation (3/8 rule)")
        log("Metrics: all channels and all test windows, train-standardized values")
        log("GPU: " + torch.cuda.get_device_name(0))
        log("PyTorch: " + torch.__version__)

        for dataset, filename, datafile, weight_folder, width, batch_size, dropout in SETTINGS:
            spec = importlib.util.spec_from_file_location("table6_" + dataset, CODE / filename)
            source = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(source)
            source.set_seed(2026)
            _, _, loader, variables, time_features = source.load_and_split_data(
                str(DATA_ROOT / datafile), seq_len=96, pred_len=720,
                batch_size=batch_size, num_workers=0)
            model = source.CTIAODEForecaster(
                96, 720, variables, time_features, d_model=width,
                primary_idx=-1, dropout=dropout)
            checkpoint = CODE / "weights" / weight_folder / "best_model_T720.pth"
            model.load_state_dict(torch.load(str(checkpoint), map_location="cpu"))
            model.to(device).eval()
            log("\nDataset: {} | source: {} | weights: weights/{}/best_model_T720.pth".format(
                dataset, filename, weight_folder))

            for label, method, nfe in SOLVERS:
                def selected_odeint(func, initial, times, **kwargs):
                    kwargs["method"] = method
                    return odeint(func, initial, times, **kwargs)
                source.odeint = selected_odeint
                squared_sum = absolute_sum = 0.0
                elements = samples = 0
                torch.cuda.synchronize()
                started = time.perf_counter()
                with torch.no_grad():
                    for batch, (x, history_time, future_time, target) in enumerate(loader, 1):
                        pred = model(x.to(device), history_time.to(device), future_time.to(device))
                        residual = (pred.cpu() - target).numpy().astype(np.float64)
                        squared_sum += np.square(residual).sum()
                        absolute_sum += np.abs(residual).sum()
                        elements += target.numel()
                        samples += len(x)
                        if batch % 20 == 0 or batch == len(loader):
                            log("{} {} | batch {}/{} | windows {}".format(
                                dataset, label, batch, len(loader), samples))
                torch.cuda.synchronize()
                log("{} {} | MSE={:.9f} | MAE={:.9f} | NFE={} | seconds={:.2f}".format(
                    dataset, label, squared_sum / elements, absolute_sum / elements,
                    nfe, time.perf_counter() - started))
            del model
            torch.cuda.empty_cache()
        log("Finished: " + datetime.now(timezone(timedelta(hours=8))).isoformat())


if __name__ == "__main__":
    main()
