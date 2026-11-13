from pathlib import Path
SCRIPT_DIR = str(Path(__file__).resolve().parent)


import os
import random
import numpy as np
import torch

def set_seed(seed=2026):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def log_to_file(file_path, text):
    print(text)
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(text + "\n")

class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data):
        self.mean = np.mean(data, axis=0, keepdims=True)
        self.std = np.std(data, axis=0, keepdims=True)
        self.std = np.maximum(self.std, 1e-5)

    def transform(self, data):
        return (data - self.mean) / self.std

@torch.no_grad()
def evaluate_model(model, dataloader, device):
    model.eval()
    total_squared_error = 0.0
    total_absolute_error = 0.0
    total_elements = 0

    for x, history_time, future_time, y in dataloader:
        x = x.to(device, non_blocking=True)
        history_time = history_time.to(device, non_blocking=True)
        future_time = future_time.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        preds = model(x, history_time, future_time)

        total_squared_error += (preds - y).pow(2).sum().item()
        total_absolute_error += (preds - y).abs().sum().item()
        total_elements += y.numel()

    mse = total_squared_error / total_elements
    mae = total_absolute_error / total_elements

    return mse, mae


import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

def build_time_features(date_series):

    dt = pd.to_datetime(date_series)

    hour = dt.dt.hour.values.astype(np.float32)
    weekday = dt.dt.weekday.values.astype(np.float32)
    day = dt.dt.day.values.astype(np.float32)
    month = dt.dt.month.values.astype(np.float32)

    features = np.stack(
        [
            np.sin(2 * np.pi * hour / 24.0),
            np.cos(2 * np.pi * hour / 24.0),

            np.sin(2 * np.pi * weekday / 7.0),
            np.cos(2 * np.pi * weekday / 7.0),

            np.sin(2 * np.pi * day / 31.0),
            np.cos(2 * np.pi * day / 31.0),

            np.sin(2 * np.pi * month / 12.0),
            np.cos(2 * np.pi * month / 12.0),
        ],
        axis=1
    )

    return features.astype(np.float32)


class MultivariateTimeSeriesDataset(Dataset):

    def __init__(
        self,
        values,
        time_features,
        seq_len,
        pred_len
    ):
        self.values = values.astype(np.float32)
        self.time_features = time_features.astype(np.float32)

        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):

        return (
            len(self.values)
            - self.seq_len
            - self.pred_len
            + 1
        )

    def __getitem__(self, idx):



        x = self.values[
            idx:
            idx + self.seq_len
        ]



        history_time = self.time_features[
            idx:
            idx + self.seq_len
        ]



        future_time = self.time_features[
            idx + self.seq_len:
            idx + self.seq_len + self.pred_len
        ]



        y = self.values[
            idx + self.seq_len:
            idx + self.seq_len + self.pred_len
        ]


        x = torch.from_numpy(x).transpose(0, 1)

        history_time = torch.from_numpy(
            history_time
        ).transpose(0, 1)

        future_time = torch.from_numpy(
            future_time
        ).transpose(0, 1)

        y = torch.from_numpy(y).transpose(0, 1)

        return x, history_time, future_time, y


def load_and_split_data(
    csv_path,
    seq_len,
    pred_len,
    batch_size=256,
    num_workers=0
):

    df = pd.read_csv(csv_path)



    possible_date_columns = [
        "date",
        "Date",
        "datetime",
        "Datetime",
        "timestamp",
        "Timestamp"
    ]

    date_col = None

    for col in possible_date_columns:

        if col in df.columns:

            date_col = col
            break

    if date_col is None:

        raise ValueError(
            "Not Find"
            "support date / Date / datetime / timestamp"
        )


    time_features = build_time_features(
        df[date_col]
    )



    value_df = df.drop(
        columns=[date_col]
    )

    values = value_df.values.astype(
        np.float32
    )

    num_vars = values.shape[1]

    dataset_name = os.path.basename(
        csv_path
    ).split(".")[0].lower()

    total_len = len(values)



    if "etth" in dataset_name:
        train_end = 12 * 30 * 24
        val_end = 16 * 30 * 24
        test_end = 20 * 30 * 24

    elif "ettm" in dataset_name:
        train_end = 12 * 30 * 24 * 4
        val_end = 16 * 30 * 24 * 4
        test_end = 20 * 30 * 24 * 4

    else:

        train_end = int(
            total_len * 0.7
        )

        val_end = int(
            total_len * 0.8
        )

        test_end = total_len




    scaler = StandardScaler()

    scaler.fit(
        values[:train_end]
    )

    values_scaled = scaler.transform(
        values
    )



    train_values = values_scaled[
        :train_end
    ]

    train_time = time_features[
        :train_end
    ]



    val_start = max(
        0,
        train_end - seq_len
    )

    val_values = values_scaled[
        val_start:
        val_end
    ]

    val_time = time_features[
        val_start:
        val_end
    ]



    test_start = max(
        0,
        val_end - seq_len
    )

    test_values = values_scaled[
        test_start:
        test_end
    ]

    test_time = time_features[
        test_start:
        test_end
    ]



    train_dataset = MultivariateTimeSeriesDataset(
        train_values,
        train_time,
        seq_len,
        pred_len
    )

    val_dataset = MultivariateTimeSeriesDataset(
        val_values,
        val_time,
        seq_len,
        pred_len
    )

    test_dataset = MultivariateTimeSeriesDataset(
        test_values,
        test_time,
        seq_len,
        pred_len
    )



    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    print(
        f"Variables: {num_vars}"
    )

    print(
        f"Train samples: {len(train_dataset)}"
    )

    print(
        f"Val samples: {len(val_dataset)}"
    )

    print(
        f"Test samples: {len(test_dataset)}"
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        num_vars,
        time_features.shape[1]
    )


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint

class RevIN(nn.Module):

    def __init__(
        self,
        num_features,
        eps=1e-5,
        affine=True
    ):

        super().__init__()

        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        if affine:

            self.affine_weight = nn.Parameter(
                torch.ones(num_features)
            )

            self.affine_bias = nn.Parameter(
                torch.zeros(num_features)
            )

    def forward(
        self,
        x,
        mode
    ):

        if mode == "norm":

            self.mean = x.mean(
                dim=-1,
                keepdim=True
            ).detach()

            self.stdev = torch.sqrt(
                x.var(
                    dim=-1,
                    keepdim=True,
                    unbiased=False
                )
                +
                self.eps
            ).detach()

            x = (
                x - self.mean
            ) / self.stdev

            if self.affine:

                x = (
                    x
                    *
                    self.affine_weight.view(
                        1,
                        -1,
                        1
                    )
                )

                x = (
                    x
                    +
                    self.affine_bias.view(
                        1,
                        -1,
                        1
                    )
                )

            return x

        elif mode == "denorm":

            if self.affine:

                x = (
                    x
                    -
                    self.affine_bias.view(
                        1,
                        -1,
                        1
                    )
                )

                x = x / (
                    self.affine_weight.view(
                        1,
                        -1,
                        1
                    )
                    +
                    self.eps
                )

            x = (
                x * self.stdev
            ) + self.mean

            return x

        else:

            raise ValueError(
                "mode must be norm or denorm"
            )

class VariableEncoder(nn.Module):

    def __init__(
        self,
        seq_len,
        d_model,
        dropout=0.05
    ):

        super().__init__()

        self.encoder = nn.Sequential(

            nn.Linear(
                seq_len,
                d_model * 2
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                d_model * 2,
                d_model
            ),

            nn.LayerNorm(
                d_model
            )
        )

    def forward(self, x):

        return self.encoder(x)

class AsymmetricPrimaryFusion(nn.Module):

    def __init__(
        self,
        d_model,
        dropout=0.05
    ):

        super().__init__()

        self.q_proj = nn.Linear(
            d_model,
            d_model,
            bias=False
        )

        self.k_proj = nn.Linear(
            d_model,
            d_model,
            bias=False
        )

        self.v_proj = nn.Linear(
            d_model,
            d_model,
            bias=False
        )

        self.gate = nn.Sequential(

            nn.Linear(
                d_model * 2,
                d_model
            ),

            nn.Sigmoid()
        )

        self.norm = nn.LayerNorm(
            d_model
        )

        self.dropout = nn.Dropout(
            dropout
        )

        self.scale = math.sqrt(
            d_model
        )

    def forward(
        self,
        H,
        primary_idx
    ):



        B, M, D = H.shape

        if M <= 1:

            return H

        aux_indices = [
            i
            for i in range(M)
            if i != primary_idx
        ]

        H_primary = H[
            :,
            primary_idx:
            primary_idx + 1,
            :
        ]

        H_aux = H[
            :,
            aux_indices,
            :
        ]

        Q = self.q_proj(
            H_primary
        )

        K = self.k_proj(
            H_aux
        )

        V = self.v_proj(
            H_aux
        )

        scores = torch.matmul(
            Q,
            K.transpose(-1, -2)
        ) / self.scale

        attention = F.softmax(
            scores,
            dim=-1
        )

        context = torch.matmul(
            attention,
            V
        )

        gate = self.gate(
            torch.cat(
                [
                    H_primary,
                    context
                ],
                dim=-1
            )
        )

        H_primary_new = self.norm(
            H_primary
            +
            gate
            *
            self.dropout(context)
        )

        H_out = H.clone()

        H_out[
            :,
            primary_idx:
            primary_idx + 1,
            :
        ] = H_primary_new

        return H_out

class CTIAODEFunc(nn.Module):

    def __init__(
        self,
        d_model
    ):

        super().__init__()

        self.d_model = d_model

        self.state_net = nn.Sequential(

            nn.Linear(
                d_model,
                d_model * 2
            ),

            nn.Tanh(),

            nn.Linear(
                d_model * 2,
                d_model
            )
        )

        nn.init.zeros_(self.state_net[-1].weight)
        nn.init.zeros_(self.state_net[-1].bias)

        self.q_proj = nn.Linear(
            d_model,
            d_model,
            bias=False
        )

        self.alpha = nn.Parameter(
            torch.tensor(0.1)
        )

        self.K_exo = None
        self.V_exo = None

    def update_exogenous(
        self,
        K_exo,
        V_exo
    ):

        self.K_exo = K_exo
        self.V_exo = V_exo

    def forward(
        self,
        t,
        H
    ):


        dH = self.state_net(
            H
        )


        if (
            self.K_exo is not None
            and
            self.V_exo is not None
        ):

            Q = self.q_proj(
                H
            )

            scores = torch.matmul(
                Q,
                self.K_exo.transpose(
                    -1,
                    -2
                )
            )

            scores = scores / math.sqrt(
                self.d_model
            )

            attention = F.softmax(
                scores,
                dim=-1
            )

            context = torch.matmul(
                attention,
                self.V_exo
            )

            dH = (
                dH
                +
                self.alpha * context
            )

        return dH

class LoopedStableBlock(nn.Module):

    def __init__(
        self,
        d_model,
        num_heads=4,
        max_loops=100,
        dropout=0.05
    ):

        super().__init__()

        self.max_loops = max_loops

        self.epsilon = 1e-3

        self.attn = nn.MultiheadAttention(

            embed_dim=d_model,

            num_heads=num_heads,

            dropout=dropout,

            batch_first=True
        )

        self.norm1 = nn.LayerNorm(
            d_model
        )

        self.norm2 = nn.LayerNorm(
            d_model
        )

        self.ffn = nn.Sequential(

            nn.Linear(
                d_model,
                d_model * 2
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                d_model * 2,
                d_model
            )
        )

    def forward(self, H):

        H_current = H

        for _ in range(
            self.max_loops
        ):

            attn_out, _ = self.attn(

                H_current,

                H_current,

                H_current,

                need_weights=False
            )

            H_new = self.norm1(

                H_current

                +

                attn_out
            )

            H_new = self.norm2(

                H_new

                +

                self.ffn(H_new)
            )

            diff = torch.norm(
                H_new - H_current,
                dim=-1
            ).mean()

            base = torch.norm(
                H_current,
                dim=-1
            ).mean()

            relative_diff = (
                    diff /
                    (base + 1e-6)
            )

            H_current = H_new

            if (
                    relative_diff.item()
                    < self.epsilon
            ):

                break

        return H_current



class MLPForecastHead(nn.Module):

    def __init__(
        self,
        d_model,
        pred_len,
        dropout=0.05
    ):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(
                d_model,
                d_model * 2
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                d_model * 2,
                pred_len
            )
        )


    def forward(self, H):

        return self.net(H)

class FutureTimeModule(nn.Module):

    def __init__(
        self,
        num_time_features,
        d_model
    ):

        super().__init__()

        self.time_encoder = nn.Sequential(

            nn.Linear(
                num_time_features,
                d_model
            ),

            nn.GELU(),

            nn.Linear(
                d_model,
                d_model
            )
        )

        self.variable_proj = nn.Linear(
            d_model,
            d_model,
            bias=False
        )

        self.time_proj = nn.Linear(
            d_model,
            d_model,
            bias=False
        )

        self.scale = math.sqrt(
            d_model
        )

    def forward(
        self,
        H,
        future_time
    ):



        future_time = future_time.transpose(
            1,
            2
        )



        time_hidden = self.time_encoder(
            future_time
        )

        H_var = self.variable_proj(
            H
        )

        H_time = self.time_proj(
            time_hidden
        )



        time_effect = torch.einsum(
            "bmd,btd->bmt",
            H_var,
            H_time
        )

        time_effect = (
            time_effect
            /
            self.scale
        )

        return time_effect


class MovingAverage(nn.Module):
    def __init__(self, kernel_size=25):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x):

        front = x[:, :, 0:1].repeat(1, 1, (self.kernel_size - 1) // 2)
        end = x[:, :, -1:].repeat(1, 1, (self.kernel_size - 1) // 2)
        x_pad = torch.cat([front, x, end], dim=-1)

        x_trend = self.avg(x_pad)
        x_seasonal = x - x_trend

        return x_trend, x_seasonal

class CTIAODEForecaster(nn.Module):

    def __init__(
        self,
        seq_len,
        pred_len,
        num_vars,
        num_time_features,
        d_model=128,
        primary_idx=-1,
        dropout=0.05
    ):

        super().__init__()

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_vars = num_vars
        self.d_model = d_model

        self.primary_idx = primary_idx



        self.revin = RevIN(
            num_vars
        )



        self.variable_encoder = VariableEncoder(

            seq_len=seq_len,

            d_model=d_model,

            dropout=dropout
        )



        self.asymmetric_fusion = (
            AsymmetricPrimaryFusion(

                d_model=d_model,

                dropout=dropout
            )
        )


        self.k_proj = nn.Linear(

            d_model,

            d_model,

            bias=False
        )

        self.v_proj = nn.Linear(

            d_model,

            d_model,

            bias=False
        )


        self.ode_func = CTIAODEFunc(
            d_model
        )



        self.stable_block = LoopedStableBlock(

            d_model=d_model,

            num_heads=4,

            max_loops=2,

            dropout=dropout
        )


        self.forecast_head = MLPForecastHead(

            d_model=d_model,

            pred_len=pred_len,

            dropout=dropout
        )


        self.future_time_module = (
            FutureTimeModule(

                num_time_features=num_time_features,

                d_model=d_model
            )
        )

        self.decomp = MovingAverage(kernel_size=25)
        self.linear_trend = nn.Linear(seq_len, pred_len)
        self.linear_seasonal = nn.Linear(seq_len, pred_len)


        self.beta_time = nn.Parameter(
            torch.tensor(0.05)
        )

        self.beta_linear = nn.Parameter(
            torch.tensor(1.0)
        )

    def forward(
        self,
        x,
        history_time,
        future_time
    ):

        B, M, L = x.shape


        if self.primary_idx == -1:

            primary_idx = M - 1

        else:

            primary_idx = self.primary_idx


        x_norm = self.revin(
            x,
            "norm"
        )


        H = self.variable_encoder(
            x_norm
        )


        H = self.asymmetric_fusion(
            H,
            primary_idx
        )


        aux_indices = [

            i
            for i in range(M)

            if i != primary_idx
        ]

        H_aux = H[
            :,
            aux_indices,
            :
        ]


        K_exo = self.k_proj(
            H_aux
        )

        V_exo = self.v_proj(
            H_aux
        )

        self.ode_func.update_exogenous(
            K_exo,
            V_exo
        )


        t_span = torch.linspace(

            0.0,

            1.0,

            steps=5,

            dtype=x.dtype,

            device=x.device
        )

        ode_out = odeint(

            self.ode_func,

            H,

            t_span,

            method="rk4"
        )

        H_T = ode_out[-1]


        H_stable = self.stable_block(
            H_T
        )


        nonlinear_pred = self.forecast_head(
            H_stable
        )

        time_effect = self.future_time_module(

            H_stable,

            future_time
        )

        trend, seasonal = self.decomp(x_norm)
        linear_pred = self.linear_trend(trend) + self.linear_seasonal(seasonal)


        pred_norm = (

            nonlinear_pred

            +

            self.beta_time
            *
            time_effect

            +

            self.beta_linear
            *
            linear_pred
        )


        prediction = self.revin(
            pred_norm,
            "denorm"
        )

        return prediction


import torch
import torch.nn as nn


class ORHL(nn.Module):
    """
    Outlier-Robust Hybrid Loss (ORHL)
    A robust M-estimator combining Elastic-Net-style local convergence
    with bounded Lipschitz gradients for large anomalies.
    """

    def __init__(self, delta=1.0, lambda_=5.0):
        super().__init__()
        self.delta = delta
        self.lambda_ = lambda_

    def forward(self, preds, target):
        abs_e = torch.abs(preds - target)
        mask_small = abs_e <= self.delta
        loss_small = 0.5 * (abs_e ** 2) + self.lambda_ * abs_e
        loss_large = (self.delta + self.lambda_) * abs_e - 0.5 * (self.delta ** 2)
        loss = torch.where(mask_small, loss_small, loss_large)
        return loss.mean()


import os
import torch


def train_and_evaluate(args, device, log_file_path):
    csv_path = os.path.join(args.root_path, args.data_path)
    (
        train_loader,
        val_loader,
        test_loader,
        num_vars,
        num_time_features
    ) = load_and_split_data(
        csv_path=csv_path,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        batch_size=args.batch_size
    )

    model = CTIAODEForecaster(
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        num_vars=num_vars,
        num_time_features=num_time_features,
        d_model=args.d_model,
        primary_idx=-1,
        dropout=args.dropout
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6
    )

    criterion = ORHL(delta=1.0, lambda_=5.0)

    best_val_mse = float("inf")
    patience_counter = 0

    weight_dir = os.path.join(SCRIPT_DIR, "checkpoints", args.dataset_name)
    os.makedirs(weight_dir, exist_ok=True)
    best_model_path = os.path.join(weight_dir, f"best_model_T{args.pred_len}.pth")

    log_to_file(log_file_path, (
        f"\n{'=' * 78}\n"
        f"M -> M CTIA-ODE Forecasting\n"
        f"Dataset    : {args.dataset_name}\n"
        f"Seq Len    : {args.seq_len}\n"
        f"Pred Len   : {args.pred_len}\n"
        f"Variables  : {num_vars}\n"
        f"D Model    : {args.d_model}\n"
        f"Batch Size : {args.batch_size}\n"
        f"LR         : {args.learning_rate}\n"
        f"Loss       : ORHL\n"
        f"{'=' * 78}"
    ))

    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0.0
        num_batches = 0

        for x, history_time, future_time, y in train_loader:
            x = x.to(device, non_blocking=True)
            history_time = history_time.to(device, non_blocking=True)
            future_time = future_time.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            preds = model(x, history_time, future_time)

            loss = criterion(preds, y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_train_loss += loss.item()
            num_batches += 1

        train_loss = total_train_loss / max(num_batches, 1)

        val_mse, val_mae = evaluate_model(model, val_loader, device)
        scheduler.step(val_mse)
        current_lr = optimizer.param_groups[0]["lr"]

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            status = "BEST"
        else:
            patience_counter += 1
            status = ""

        epoch_msg = (
            f"Epoch {epoch + 1:03d} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val MSE: {val_mse:.6f} | "
            f"Val MAE: {val_mae:.6f} | "
            f"LR: {current_lr:.2e} | "
            f"ES: {patience_counter}/{args.patience} {status}"
        )
        log_to_file(log_file_path, epoch_msg)

        if patience_counter >= args.patience:
            log_to_file(log_file_path, "Early stopping triggered.")
            break

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_mse, test_mae = evaluate_model(model, test_loader, device)

    final_msg = f"\nFINAL RESULT | T={args.pred_len} | MSE={test_mse:.6f} | MAE={test_mae:.6f}"
    log_to_file(log_file_path, final_msg)

    return test_mse, test_mae


import os
import argparse
import datetime
import torch
import numpy as np


def main():
    parser = argparse.ArgumentParser(description='CTIA-ODE for Multivariate Time Series Forecasting')

    parser.add_argument('--root_path', type=str, default='.', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='weather.csv', help='data file')
    parser.add_argument('--dataset_name', type=str, default='weather', help='dataset name')

    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--pred_lens', type=int, nargs='+', default=[96, 192, 336, 720],
                        help='prediction sequence lengths')

    parser.add_argument('--d_model', type=int, default=128, help='dimension of model')
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout rate')
    parser.add_argument('--epochs', type=int, default=100, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=12, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='optimizer learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')

    args = parser.parse_args()
    if not os.path.isabs(args.root_path):
        args.root_path = os.path.join(SCRIPT_DIR, args.root_path)

    set_seed(2026)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    results_dir = os.path.join(SCRIPT_DIR, "results", args.dataset_name)
    os.makedirs(results_dir, exist_ok=True)
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = os.path.join(results_dir, f"metrics_{current_time}.txt")

    with open(log_file_path, "w", encoding="utf-8") as f:
        f.write(f"Experiment: {args.dataset_name}\nTime: {current_time}\n")

    all_results = []

    for pred_len in args.pred_lens:
        args.pred_len = pred_len
        print(f">>>>>>> Start training : {args.dataset_name} | Seq: {args.seq_len} | Pred: {args.pred_len} >>>>>>>")

        mse, mae = train_and_evaluate(args, device, log_file_path)
        all_results.append((pred_len, mse, mae))

    log_to_file(log_file_path, "\n" + "=" * 78 + "\nFINAL SUMMARY\n" + "=" * 78)
    for pred_len, mse, mae in all_results:
        log_to_file(log_file_path, f"T={pred_len:<3d} | MSE={mse:.6f} | MAE={mae:.6f}")

    avg_mse = np.mean([res[1] for res in all_results])
    avg_mae = np.mean([res[2] for res in all_results])
    log_to_file(log_file_path, "-" * 78 + f"\nAverage MSE={avg_mse:.6f} | Average MAE={avg_mae:.6f}\n" + "=" * 78)


if __name__ == "__main__":
    main()