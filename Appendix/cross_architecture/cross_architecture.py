"""Weather L96 T96: DLinear/iTransformer x MSE/ORHL, seed2026."""
import importlib.util
import io
import sys
import time
import traceback
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
CODE = HERE.parent.parent
LIBRARY = CODE.parent / 'work2/third_party/Time-Series-Library'
sys.path.insert(0, str(LIBRARY))
from models import DLinear, iTransformer

DATA = Path(r'D:\all_datasets\weather\weather.csv')


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


def forward(model, batch):
    x, history, future, target = [v.cuda().transpose(1, 2).contiguous() for v in batch]
    decoder = torch.cat([x[:, -48:], torch.zeros_like(target)], dim=1)
    decoder_time = torch.cat([history[:, -48:], future], dim=1)
    return model(x, history, decoder, decoder_time), target


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    squared = absolute = 0.
    count = 0
    for batch in loader:
        prediction, target = forward(model, batch)
        error = (prediction-target).cpu().numpy().astype(np.float64)
        squared += np.square(error).sum()
        absolute += np.abs(error).sum()
        count += error.size
    return squared/count, absolute/count


def run(source, architecture, objective):
    source.set_seed(2026)
    with redirect_stdout(io.StringIO()):
        train, validation, test, channels, _ = source.load_and_split_data(
            str(DATA), seq_len=96, pred_len=96, batch_size=256, num_workers=0)
    train.sampler.generator = torch.Generator().manual_seed(2026)
    for i, loader in enumerate([train, validation, test]):
        loader.generator = torch.Generator().manual_seed(2026+10000*(i+1))
    config = SimpleNamespace(task_name='long_term_forecast', seq_len=96, pred_len=96,
        enc_in=channels, dec_in=channels, c_out=channels, moving_avg=25,
        d_model=512, d_ff=512, e_layers=3, n_heads=8, factor=3,
        embed='timeF', freq='t', dropout=.1, activation='gelu')
    model = (DLinear if architecture=='DLinear' else iTransformer).Model(config).cuda()
    criterion = nn.MSELoss() if objective=='MSE' else ORHL(delta=1.0, lambda_=5.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=.5, patience=3, min_lr=1e-6)
    print('START architecture={} objective={} | {} parameters={}'.format(
        architecture, objective, datetime.now().isoformat(), sum(p.numel() for p in model.parameters())), flush=True)
    best, best_state, best_epoch, stale = float('inf'), None, 0, 0
    started = time.perf_counter()
    max_norm = 0.
    steps = clipped = 0
    for epoch in range(1, 101):
        epoch_started = time.perf_counter()
        model.train()
        for batch in train:
            optimizer.zero_grad(set_to_none=True)
            prediction, target = forward(model, batch)
            loss = criterion(prediction, target)
            if not torch.isfinite(loss).item():
                raise RuntimeError('Non-finite objective: {} {}'.format(architecture,objective))
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(),5.).item()
            if not np.isfinite(norm):
                raise RuntimeError('Non-finite gradient: {} {}'.format(architecture,objective))
            max_norm = max(max_norm,norm)
            steps += 1
            clipped += int(norm>5.)
            optimizer.step()
        val_mse,val_mae = evaluate(model,validation)
        if not np.isfinite(val_mse):
            raise RuntimeError('Non-finite validation MSE')
        scheduler.step(val_mse)
        if val_mse<best:
            best,best_epoch,stale = val_mse,epoch,0
            best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            stale += 1
        print('{} {} Epoch={:03d} val_MSE={:.9f} val_MAE={:.9f} ES={}/12 lr={:.2e} seconds={:.2f}'.format(
            architecture,objective,epoch,val_mse,val_mae,stale,
            optimizer.param_groups[0]['lr'],time.perf_counter()-epoch_started),flush=True)
        if stale>=12:
            break
    model.load_state_dict(best_state)
    mse,mae = evaluate(model,test)
    print('RESULT architecture={} objective={} | MSE={:.9f} MAE={:.9f} best_epoch={} epochs={} seconds={:.2f}'.format(
        architecture,objective,mse,mae,best_epoch,epoch,time.perf_counter()-started),flush=True)
    print('GRADIENT preclip_max={:.6f} clipping_fraction={:.6f}'.format(max_norm,clipped/steps),flush=True)
    del model,optimizer,scheduler,best_state,prediction,loss
    torch.cuda.empty_cache()
    return mse,mae


def main():
    sys.dont_write_bytecode=True
    torch.set_num_threads(2)
    spec=importlib.util.spec_from_file_location('cross_weather',CODE/'weather.py')
    source=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(source)
    print('Cross-architecture objective comparison | '+datetime.now().isoformat(),flush=True)
    print('Weather L96 T96 seed2026; 4 fresh baseline runs. CTIA-ODE NOT run or assigned a result here.',flush=True)
    print('Implementation: '+str(LIBRARY),flush=True)
    print('DLinear: shared channel weights, moving_avg25. iTransformer: d512 ff512 layers3 heads8 dropout.1.',flush=True)
    print('Original baseline architectures; same source Weather split/scaler/channel order and eight source calendar features. DLinear ignores calendar features by design.',flush=True)
    print('Both arms per architecture share initialization and independent loader seeds. Only objective changes.',flush=True)
    print('MSE or Huber(delta1)+5*MAE; fixed settings, no hyperparameter search; no contamination.',flush=True)
    print('ORHL uses the author-provided piecewise ORHL(nn.Module), delta=1.0, lambda_=5.0.',flush=True)
    print('AdamW lr5e-4 wd1e-4 batch256 FP32 norm5; max_epochs100 early_stop12; plateau factor.5 patience3 min_lr1e-6.',flush=True)
    print('Best validation MSE checkpoint; all-channel, all-test-window standardized metrics. Single seed, no cross-seed SD. This is a new fixed-protocol comparison, not reproduction of Table1.',flush=True)
    results=[]
    for architecture in ['DLinear','iTransformer']:
        for objective in ['MSE','ORHL']:
            mse,mae=run(source,architecture,objective)
            results.append((architecture,objective,mse,mae))
    print('FINAL RESULTS',flush=True)
    for architecture,objective,mse,mae in results:
        print('{} {} | MSE={:.9f} MAE={:.9f}'.format(architecture,objective,mse,mae),flush=True)
    print('Finished: '+datetime.now().isoformat(),flush=True)


if __name__=='__main__':
    with (HERE/'cross_architecture.log').open('x',encoding='utf-8',buffering=1) as output:
        with redirect_stdout(output),redirect_stderr(output):
            try:
                main()
            except BaseException:
                traceback.print_exc()
                raise
