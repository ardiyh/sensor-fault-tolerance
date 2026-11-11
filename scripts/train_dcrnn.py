"""
Paper 7 copy of paper2_STC/STC-code/scripts/train_dcrnn.py. Changes are marked "Paper 7":
the model comes from src/dcrnn_fixed.py (diffusion actually applied), data/utils come from the
read-only Paper 2 tree via src.stc_bridge, and checkpoints/results are written under SFT-code.

Train DCRNN Teacher Model.

Usage:
    python -u train_dcrnn.py --dataset metr-la
    python -u train_dcrnn.py --dataset pems-bay --hidden 64 --layers 2
"""

import torch
import torch.nn as nn
import numpy as np
import os
import sys
import argparse
import time
import math
import json
from torch.cuda.amp import GradScaler, autocast

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Paper 7: SFT-code root
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
from src import stc_bridge  # Paper 7: Paper 2 utils on sys.path, read-only
stc_bridge._ensure_on_path()

from torch.utils.data import DataLoader
from src.dcrnn_fixed import DCRNNTeacher  # Paper 7: fixed diffusion step
from utils.data_loader import load_traffic_data, SlidingWindowDataset, get_scaler, NORM_RECOMMENDATION
from utils.checkpoint_tags import build_dcrnn_checkpoint_name
from utils.common import compute_metrics, compute_mase


def train(args):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("Training DCRNN Teacher")
    print("=" * 60)
    print(f"  Dataset:      {args.dataset}")
    print(f"  Device:       {DEVICE}")
    print(f"  Epochs:       {args.epochs}")
    print(f"  Batch size:   {args.batch_size}")
    print(f"  LR:           {args.lr}")
    print(f"  Hidden:       {args.hidden}")
    print(f"  Layers:       {args.layers}")
    print(f"  Diffusion K:  {args.K}")
    print(f"  Normalize:    {args.normalize} (per-node={args.per_node_norm})")
    print(f"  Num workers:  {args.num_workers}")
    print("=" * 60)

    use_amp = DEVICE == "cuda"

    # =========================================================
    # 1. Load Data
    # =========================================================
    print("\n[1/4] Loading data...")
    data, edge_index, edge_weight, timestamps = load_traffic_data(args.dataset)
    num_nodes = data.shape[1]
    edge_index = edge_index.to(DEVICE)

    n = data.shape[0]
    n_train = int(n * 0.7)
    n_val = int(n * 0.1)

    scaler = None
    if args.normalize:
        print("  Applying StandardScaler (fitted on train split only)...")
        scaler = get_scaler(data, train_ratio=0.7, per_node=args.per_node_norm)
        data = scaler.transform(data)

    train_data = data[:n_train]
    val_data = data[n_train:n_train + n_val]

    train_dataset = SlidingWindowDataset(train_data, input_len=12, output_len=12)
    val_dataset = SlidingWindowDataset(val_data, input_len=12, output_len=12)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    print(f"  Data shape: {data.shape}")
    print(f"  Train windows: {len(train_dataset)}")
    print(f"  Val windows:   {len(val_dataset)}")

    # =========================================================
    # 2. Build Model
    # =========================================================
    print("\n[2/4] Building model...")
    model = DCRNNTeacher(
        num_nodes=num_nodes,
        in_channels=1,
        hidden_channels=args.hidden,
        out_channels=1,
        num_layers=args.layers,
        K=args.K,
        prediction_length=12,
        dropout=args.dropout,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")

    # =========================================================
    # 3. Training Loop
    # =========================================================
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    criterion = nn.MSELoss()
    amp_scaler = GradScaler(enabled=use_amp)

    os.makedirs(os.path.join(BASE_DIR, "checkpoints"), exist_ok=True)  # Paper 7
    checkpoint_path = os.path.join(  # Paper 7: dcrnn_fixed_<ds>[_tag].pth under SFT-code/checkpoints
        BASE_DIR, "checkpoints", build_dcrnn_checkpoint_name(args.dataset, args.checkpoint_tag).replace("dcrnn_", "dcrnn_fixed_", 1)
    )
    best_val_loss = float("inf")
    patience_counter = 0

    print(f"\n[3/4] Training for {args.epochs} epochs (AMP={use_amp})...")

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        num_batches = 0
        epoch_start = time.time()

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()
            with autocast(enabled=use_amp):
                pred = model(x, edge_index)
                loss = criterion(pred, y)
            amp_scaler.scale(loss).backward()
            amp_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            amp_scaler.step(optimizer)
            amp_scaler.update()

            train_loss += loss.item()
            num_batches += 1

            if (batch_idx + 1) % 100 == 0:
                print(f"  Epoch {epoch+1} | Batch {batch_idx+1}/{len(train_loader)} | "
                      f"Loss: {loss.item():.6f}")

        avg_train_loss = train_loss / num_batches

        # Validate
        model.eval()
        val_loss = 0.0
        val_batches = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
                with autocast(enabled=use_amp):
                    pred = model(x, edge_index)
                    # Mask out nan/inf targets (e.g. per-node-norm with near-zero std)
                    finite_mask = torch.isfinite(y) & torch.isfinite(pred)
                    if finite_mask.sum() == 0:
                        continue
                    loss = criterion(pred[finite_mask], y[finite_mask])
                if torch.isfinite(loss):
                    val_loss += loss.item()
                    val_batches += 1
                all_preds.append(pred.float().cpu())
                all_targets.append(y.float().cpu())

        avg_val_loss = val_loss / val_batches if val_batches > 0 else float("nan")
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        finite_mask = torch.isfinite(all_targets) & torch.isfinite(all_preds)
        mae = torch.mean(torch.abs(all_preds[finite_mask] - all_targets[finite_mask])).item() \
              if finite_mask.sum() > 0 else float("nan")

        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Train Loss: {avg_train_loss:.6f} | "
              f"Val Loss: {avg_val_loss:.6f} | "
              f"Val MAE: {mae:.4f} | "
              f"LR: {current_lr:.2e} | "
              f"Time: {epoch_time:.1f}s")

        # Use MAE when loss is nan (e.g. per-node-norm overflow on ECL)
        monitor = mae if (args.normalize and math.isnan(avg_val_loss)) else avg_val_loss
        scheduler.step(monitor if not math.isnan(monitor) else 0.0)

        if monitor < best_val_loss:
            best_val_loss = monitor
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  → Saved best model (val_mae={best_val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    # =========================================================
    # 4. Final Evaluation
    # =========================================================
    print(f"\n[4/4] Final evaluation...")
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE, weights_only=True))
    model.eval()

    test_data = data[n_train + n_val:]
    test_dataset = SlidingWindowDataset(test_data, input_len=12, output_len=12)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            with autocast(enabled=use_amp):
                pred = model(x, edge_index)
            all_preds.append(pred.cpu())
            all_targets.append(y.cpu())

    all_preds = torch.cat(all_preds, dim=0).float().numpy()
    all_targets = torch.cat(all_targets, dim=0).float().numpy()

    if scaler is not None:
        all_preds = scaler.inverse_transform(all_preds)
        all_targets = scaler.inverse_transform(all_targets)

    train_true = data[:n_train]
    if scaler is not None:
        train_true = scaler.inverse_transform(train_true)
    train_true = train_true.squeeze(-1) if train_true.ndim == 3 and train_true.shape[-1] == 1 else train_true

    mae, rmse, mape, nmae = compute_metrics(all_preds, all_targets)
    mase = compute_mase(mae, train_true, m=1)

    print("=" * 60)
    print("FINAL TEST RESULTS (DCRNN)")
    print("=" * 60)
    print(f"  Dataset: {args.dataset}")
    print(f"  MAE:     {mae:.4f}")
    print(f"  RMSE:    {rmse:.4f}")
    print(f"  MAPE:    {mape:.2f}%")
    print(f"  nMAE:    {nmae:.4f}")
    print(f"  MASE:    {mase:.4f}")
    print(f"  Params:  {trainable_params:,}")
    print(f"  Saved:   {checkpoint_path}")
    print("=" * 60)

    if args.results_json:
        result = {
            "name": "DCRNN (fixed diffusion, Paper 7)",
            "dataset": args.dataset,
            "mae": float(mae),
            "rmse": float(rmse),
            "mape": float(mape),
            "nmae": float(nmae),
            "mase": None if np.isnan(mase) else float(mase),
            "params": int(trainable_params),
            "epochs_requested": int(args.epochs),
            "batch_size": int(args.batch_size),
            "checkpoint": checkpoint_path,
        }
        output_path = os.path.join(BASE_DIR, args.results_json)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump([result], f, indent=2)
            f.write("\n")
        print(f"  Results JSON: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DCRNN Teacher")
    parser.add_argument("--dataset", type=str, default="metr-la",
                        choices=["metr-la", "pems-bay", "solar-al", "sdwpf", "acn-ev", "pems04", "pems03", "ecl", "gefcom-wind", "opsd-wind", "opsd-solar", "ercot"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=64,
                        help="Hidden channels in DCGRU cells")
    parser.add_argument("--layers", type=int, default=2,
                        help="Number of DCGRU layers")
    parser.add_argument("--K", type=int, default=2,
                        help="Diffusion steps")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=15,
                        help="Early stopping patience")
    parser.add_argument("--checkpoint-tag", type=str, default=None,
                        help="Optional suffix for saved DCRNN checkpoint filename")
    parser.add_argument("--results-json", type=str, default=None,
                        help="Optional output path for final test metrics JSON")
    parser.add_argument("--normalize", action="store_true",
                        help="Apply Z-score normalization (fitted on train split)")
    parser.add_argument("--per-node-norm", action="store_true",
                        help="Use per-node Z-score (recommended for heterogeneous nodes)")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader num_workers")

    args = parser.parse_args()
    train(args)
