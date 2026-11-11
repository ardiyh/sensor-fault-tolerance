"""
Paper 7 copy of paper2_STC/STC-code/scripts/train_distill_attention.py. Changes are marked "Paper 7":
Paper 2 utils/models/data are read from the read-only Paper 2 tree via src.stc_bridge,
student checkpoints are written to SFT-code/checkpoints, and --sensor-dropout adds the
augmentation in src/augment.py. Training logic is otherwise unchanged.
"""
"""
Knowledge Distillation with Attention Transfer (AT-KD).

Trains Student-Attention model using pre-computed Teacher outputs.
Supports Teacher v1 and v2 cache files.
Also supports --baseline mode (no KD) for ablation studies.

Usage:
    python -u train_distill_attention.py --dataset metr-la
    python -u train_distill_attention.py --dataset metr-la --teacher-version v2
    python -u train_distill_attention.py --dataset metr-la --teacher-type dcrnn
    python -u train_distill_attention.py --dataset metr-la --baseline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
import argparse
import time

SFT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Paper 7: SFT-code root
if SFT_DIR not in sys.path:
    sys.path.insert(0, SFT_DIR)
from src import stc_bridge  # Paper 7
from src.augment import sensor_dropout, zero_fill_value  # Paper 7
BASE_DIR = str(stc_bridge.stc_root())  # Paper 7: Paper 2 tree (data, utils), read-only
stc_bridge._ensure_on_path()
CKPT_DIR = os.path.join(SFT_DIR, "checkpoints")  # Paper 7: where students are saved
os.makedirs(CKPT_DIR, exist_ok=True)

from torch.utils.data import DataLoader, TensorDataset
from models.student_attention import AttentionST
from utils.data_loader import load_traffic_data, SlidingWindowDataset, DATASETS, get_scaler, NORM_RECOMMENDATION
from utils.common import compute_metrics
from utils.checkpoint_tags import build_student_attention_checkpoint_name, build_teacher_cache_name

# A100 optimisation: auto-tunes conv/matmul kernels after first batch
torch.backends.cudnn.benchmark = True


def load_teacher_cache(
    dataset,
    teacher_version="v1",
    teacher_type="chronos",
    cache_tag=None,
    teacher_source="precompute",
    chronos2_cache_path=None,
    base_dir=None,
):
    """Load pre-computed teacher outputs from precompute or Chronos2 cache."""
    project_root = base_dir if base_dir is not None else BASE_DIR

    if teacher_source == "chronos2":
        if chronos2_cache_path:
            cache_path = chronos2_cache_path
        else:
            cache_path = os.path.join(project_root, "results", "tsfm", f"cache_chronos2_{dataset}_c168_trainvaltest.pt")

        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Chronos2 cache not found: {cache_path}\n"
                f"Run cache_chronos2_forecasts.py first for dataset={dataset}."
            )

        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        if "splits" not in cache or "train" not in cache["splits"] or "val" not in cache["splits"]:
            raise ValueError(f"Invalid Chronos2 cache format: {cache_path}")

        print(f"  Loaded Chronos2 cache: {cache_path}")
        print(f"  Train forecast: {cache['splits']['train']['forecast'].shape}")
        print(f"  Val forecast:   {cache['splits']['val']['forecast'].shape}")
        cache["source"] = "chronos2"
        return cache

    ds_info = DATASETS[dataset]
    cache_dir = os.path.join(project_root, "data", ds_info["data_subdir"])

    cache_name = build_teacher_cache_name(
        dataset,
        teacher_type=teacher_type,
        teacher_version=teacher_version,
        cache_tag=cache_tag,
    )

    cache_path = os.path.join(cache_dir, cache_name)

    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"Teacher cache not found: {cache_path}\n"
            f"Run precompute_teacher.py first:\n"
            f"  python precompute_teacher.py --dataset {dataset} --teacher-version {teacher_version}"
        )

    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    print(f"  Loaded teacher cache: {cache_path}")
    print(f"  Teacher preds: {cache['teacher_preds'].shape}")
    print(f"  Teacher attns: {cache['teacher_attns'].shape}")
    cache["source"] = "precompute"
    return cache


def attention_transfer_loss(student_attn, teacher_attn):
    """
    Compute AT loss between student and teacher attention maps.
    Uses L2 norm of difference between attention maps (normalized).
    """
    s_norm = F.normalize(student_attn.view(student_attn.size(0), -1), dim=1)
    t_norm = F.normalize(teacher_attn.view(teacher_attn.size(0), -1), dim=1)
    return (s_norm - t_norm).pow(2).mean()


def train(args):
    import random
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # BF16 on A100: hardware-native, no loss-scaling needed
    use_amp = DEVICE == "cuda"
    amp_dtype = torch.bfloat16 if use_amp else torch.float32

    print("=" * 60)
    if args.baseline:
        print("BASELINE Training: Student-Attention (no KD)")
    else:
        print("AT-KD Training: Student-Attention")
    print("=" * 60)
    print(f"  Dataset:         {args.dataset}")
    print(f"  Device:          {DEVICE}")
    print(f"  Mode:            {'BASELINE (no KD)' if args.baseline else 'KD + AT'}")
    if not args.baseline:
        print(f"  Teacher version: {args.teacher_version}")
        print(f"  Teacher type:    {args.teacher_type}")
        print(f"  Teacher source:  {args.teacher_source}")
        print(f"  Alpha (output):  {args.alpha}")
        print(f"  Beta (attention):{args.beta}")
    print(f"  Normalize:       {args.normalize} (per-node={args.per_node_norm})")
    print(f"  AMP dtype:       {amp_dtype}")
    print("=" * 60)

    # =========================================================
    # 1. Load Data + Teacher Cache
    # =========================================================
    print("\n[1/4] Loading data...")
    data, edge_index, edge_weight, timestamps = load_traffic_data(args.dataset)
    num_nodes = data.shape[1]
    edge_index = edge_index.to(DEVICE)

    if args.normalize:
        print("  Normalizing data using StandardScaler (fitted on train split only)...")
        scaler = get_scaler(data, train_ratio=0.7, per_node=args.per_node_norm)
        data = scaler.transform(data)
    else:
        scaler = None

    L = data.shape[0]
    train_end = int(L * 0.7)
    val_end = int(L * 0.8)

    train_data = data[:train_end]
    val_data = data[train_end:val_end]

    train_dataset = SlidingWindowDataset(train_data, input_len=12, output_len=12)
    val_dataset = SlidingWindowDataset(val_data, input_len=12, output_len=12)

    n_train = len(train_dataset)
    n_val = len(val_dataset)

    train_x = torch.stack([train_dataset[i][0] for i in range(n_train)])
    train_y = torch.stack([train_dataset[i][1] for i in range(n_train)])
    val_x = torch.stack([val_dataset[i][0] for i in range(n_val)])
    val_y = torch.stack([val_dataset[i][1] for i in range(n_val)])

    dl_kwargs = dict(
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=4 if args.num_workers > 0 else None,
    )

    if args.baseline:
        print("  BASELINE MODE: No teacher cache loaded.")
        train_dataset = TensorDataset(train_x, train_y)
    else:
        print("  Loading teacher cache...")
        cache = load_teacher_cache(
            args.dataset,
            args.teacher_version,
            args.teacher_type,
            cache_tag=args.cache_tag,
            teacher_source=args.teacher_source,
            chronos2_cache_path=args.chronos2_cache_path,
        )
        if cache.get("source") == "chronos2":
            train_teacher_preds = cache["splits"]["train"]["forecast"].float()
            if args.normalize and scaler is not None:
                train_teacher_preds = scaler.transform(train_teacher_preds)
            min_n = min(train_teacher_preds.shape[0], train_x.shape[0])
            if train_teacher_preds.shape[0] != train_x.shape[0]:
                print(f"  WARNING: train cache windows mismatch ({train_teacher_preds.shape[0]} vs {train_x.shape[0]}), truncating to {min_n}")
            train_dataset = TensorDataset(train_x[:min_n], train_y[:min_n], train_teacher_preds[:min_n])
        else:
            teacher_preds = cache["teacher_preds"]
            teacher_attns = cache["teacher_attns"]
            train_teacher_preds = teacher_preds[:n_train]
            train_teacher_attns = teacher_attns[:n_train]
            train_dataset = TensorDataset(train_x, train_y, train_teacher_preds, train_teacher_attns)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **dl_kwargs)
    val_dataset = TensorDataset(val_x, val_y)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, **dl_kwargs)

    print(f"  Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    # =========================================================
    # 2. Build Student
    # =========================================================
    print("\n[2/4] Building Student-Attention model...")
    student = AttentionST(
        num_nodes=num_nodes,
        in_steps=12,
        out_steps=12,
        channels=32,
        n_heads=4,
    ).to(DEVICE)
    if not args.no_compile:
        torch._dynamo.config.suppress_errors = True  # fallback to eager if Triton compile fails
        student = torch.compile(student)  # JIT-compile for A100 throughput
        print(f"  torch.compile: applied")
    else:
        print(f"  torch.compile: disabled (eager mode)")

    student_params = sum(p.numel() for p in student.parameters())
    print(f"  Student params: {student_params:,}")

    # =========================================================
    # 3. Training Loop
    # =========================================================
    optimizer = torch.optim.Adam(student.parameters(), lr=args.lr)
    # Instantiate loss function
    loss_name = args.loss.lower()
    if loss_name == "mae":
        task_criterion = nn.L1Loss()
    elif loss_name == "huber":
        task_criterion = nn.HuberLoss(delta=1.0)
    else:
        task_criterion = nn.MSELoss()
    print(f"  Loss:         {loss_name.upper()}")
    print(f"  AMP (bfloat16): {'enabled' if use_amp else 'disabled'}")

    ckpt_name = build_student_attention_checkpoint_name(
        args.dataset,
        baseline=args.baseline,
        teacher_type=args.teacher_type,
        teacher_version=args.teacher_version,
        checkpoint_tag=args.checkpoint_tag,
    )
    ckpt_path = os.path.join(CKPT_DIR, ckpt_name)  # Paper 7

    best_val_loss = float("inf")
    patience_counter = 0

    print(f"\n[3/4] Training for {args.epochs} epochs...")

    for epoch in range(args.epochs):
        student.train()
        train_loss = 0.0
        num_batches = 0
        epoch_start = time.time()

        for batch_idx, batch in enumerate(train_loader):
            if args.baseline:
                x, y = batch
                x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            else:
                if len(batch) == 4:
                    x, y, t_pred, t_attn = batch
                else:
                    x, y, t_pred = batch
                    t_attn = None
                x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
                t_pred = t_pred.to(DEVICE, non_blocking=True)
                if t_attn is not None:
                    t_attn = t_attn.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()
            if args.sensor_dropout > 0:  # Paper 7
                x = sensor_dropout(x, args.sensor_dropout, zero_fill_value(scaler))

            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                if args.baseline:
                    s_pred = student(x, edge_index)
                    if isinstance(s_pred, tuple):
                        s_pred = s_pred[0]
                    loss = task_criterion(s_pred, y)
                else:
                    s_pred, s_attn = student(x, edge_index, return_attention=True)

                    loss_task = task_criterion(s_pred, y)
                    loss_output = task_criterion(s_pred, t_pred)

                    if s_attn is not None and t_attn is not None:
                        t_attn_avg = t_attn.mean(dim=1, keepdim=True)
                        s_attn_avg = s_attn.mean(dim=1, keepdim=True)
                        loss_attn = attention_transfer_loss(s_attn_avg, t_attn_avg)
                    else:
                        loss_attn = torch.tensor(0.0, device=DEVICE)

                    loss = loss_task + args.alpha * loss_output + args.beta * loss_attn

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

            if (batch_idx + 1) % 100 == 0:
                if args.baseline:
                    print(f"  Epoch {epoch+1} | Batch {batch_idx+1}/{len(train_loader)} | "
                          f"Loss={loss.item():.4f}")
                else:
                    print(f"  Epoch {epoch+1} | Batch {batch_idx+1}/{len(train_loader)} | "
                          f"L_task={loss_task.item():.4f} L_out={loss_output.item():.4f} "
                          f"L_attn={loss_attn.item():.4f}")

        avg_train_loss = train_loss / num_batches

        # Validate
        student.eval()
        val_loss = 0.0
        val_batches = 0
        all_preds, all_targets = [], []

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            for x, y in val_loader:
                x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
                pred = student(x, edge_index)
                if isinstance(pred, tuple):
                    pred = pred[0]
                loss = task_criterion(pred, y)
                val_loss += loss.item()
                val_batches += 1
                all_preds.append(pred.cpu())
                all_targets.append(y.cpu())

        avg_val_loss = val_loss / val_batches
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        if args.normalize and scaler is not None:
            all_preds = scaler.inverse_transform(all_preds)
            all_targets = scaler.inverse_transform(all_targets)

        mae = torch.mean(torch.abs(all_preds - all_targets)).item()

        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Train: {avg_train_loss:.6f} | "
              f"Val: {avg_val_loss:.6f} | "
              f"MAE: {mae:.4f} | "
              f"Time: {epoch_time:.1f}s")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(student.state_dict(), ckpt_path)
            print(f"  → Saved best model (val_loss={best_val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    # =========================================================
    # 4. Final Test Evaluation
    # =========================================================
    print(f"\n[4/4] Final evaluation...")
    student.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
    student.eval()

    n_test_start = int(data.shape[0] * 0.8)
    test_data = data[n_test_start:]
    test_dataset = SlidingWindowDataset(test_data, input_len=12, output_len=12)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, **dl_kwargs
    )

    all_preds, all_targets = [], []
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
        for x, y in test_loader:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            pred = student(x, edge_index)
            if isinstance(pred, tuple):
                pred = pred[0]
            all_preds.append(pred.cpu())
            all_targets.append(y.cpu())

    all_preds = torch.cat(all_preds, dim=0).float().numpy()
    all_targets = torch.cat(all_targets, dim=0).float().numpy()

    if args.normalize and scaler is not None:
        all_preds = scaler.inverse_transform(all_preds)
        all_targets = scaler.inverse_transform(all_targets)

    mae, rmse, mape, nmae = compute_metrics(all_preds, all_targets)

    print("=" * 60)
    if args.baseline:
        print(f"FINAL TEST RESULTS (Student-Attn, BASELINE)")
    else:
        print(f"FINAL TEST RESULTS (Student-Attn, teacher={args.teacher_type} {args.teacher_version})")
    print("=" * 60)
    print(f"  Dataset:     {args.dataset}")
    print(f"  Seed:        {args.seed}")
    print(f"  MAE:         {mae:.4f}")
    print(f"  RMSE:        {rmse:.4f}")
    print(f"  MAPE:        {mape:.2f}%")
    print(f"  nMAE:        {nmae:.4f}")
    print(f"  Params:      {student_params:,}")
    print(f"  Saved:       {ckpt_path}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AT-KD Training")
    parser.add_argument("--dataset", type=str, default="metr-la",
                        choices=["metr-la", "pems-bay", "solar-al", "sdwpf", "acn-ev", "pems04", "pems03", "ecl", "gefcom-wind", "opsd-wind", "opsd-solar", "ercot"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size (A100 default: 256)")
    parser.add_argument("--loss", type=str, default="mse", choices=["mse", "mae", "huber"],
                        help="Loss function: mse, mae, or huber")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--baseline", action="store_true",
                        help="Train Student-Attention from scratch (no KD/AT). "
                             "For ablation: architecture-only vs KD+AT.")

    # KD hyperparameters
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Weight for output distillation loss")
    parser.add_argument("--beta", type=float, default=0.1,
                        help="Weight for attention transfer loss")

    # Teacher selection
    parser.add_argument("--teacher-version", type=str, default="v1",
                        choices=["v1", "v2"])
    parser.add_argument("--teacher-type", type=str, default="chronos",
                        choices=["chronos", "dcrnn"],
                        help="Teacher model type")
    parser.add_argument("--teacher-source", type=str, default="precompute",
                        choices=["precompute", "chronos2"],
                        help="Teacher cache source")
    parser.add_argument("--chronos2-cache-path", type=str, default=None,
                        help="Explicit path to Chronos2 cache .pt file")
    parser.add_argument("--cache-tag", type=str, default=None,
                        help="Optional suffix for loading teacher cache filename")
    parser.add_argument("--checkpoint-tag", type=str, default=None,
                        help="Optional suffix for saved student checkpoint filename")
    parser.add_argument("--normalize", action="store_true",
                        help="Apply Z-score normalization")
    parser.add_argument("--per-node-norm", action="store_true",
                        help="Use per-node Z-score (recommended for heterogeneous nodes)")
    parser.add_argument("--no-compile", action="store_true",
                        help="Disable torch.compile (use eager mode)")
    parser.add_argument("--num-workers", type=int, default=8,
                        help="DataLoader num_workers (DGX A100 default: 8)")
    parser.add_argument("--sensor-dropout", type=float, default=0.0,  # Paper 7
                        help="Max per-sample sensor-dropout rate (0 = off); see src/augment.py")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")

    args = parser.parse_args()
    train(args)
