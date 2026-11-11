"""
Paper 7 copy of paper2_STC/STC-code/scripts/train_distill.py. Changes are marked "Paper 7":
Paper 2 utils/models/data are read from the read-only Paper 2 tree via src.stc_bridge,
student checkpoints are written to SFT-code/checkpoints, and --sensor-dropout adds the
augmentation in src/augment.py. Training logic is otherwise unchanged.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import sys
import copy

SFT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Paper 7: SFT-code root
if SFT_DIR not in sys.path:
    sys.path.insert(0, SFT_DIR)
from src import stc_bridge  # Paper 7
from src.augment import sensor_dropout, zero_fill_value  # Paper 7
BASE_DIR = str(stc_bridge.stc_root())  # Paper 7: Paper 2 tree (data, utils), read-only
stc_bridge._ensure_on_path()
CKPT_DIR = os.path.join(SFT_DIR, "checkpoints")  # Paper 7: where students are saved
os.makedirs(CKPT_DIR, exist_ok=True)

from models.teacher import STChronosTeacher
from models.student import LightweightST
from utils.data_loader import load_traffic_data, SlidingWindowDataset, DATASETS, get_scaler, NORM_RECOMMENDATION
from utils.common import remap_legacy_teacher_keys, compute_metrics
from utils.checkpoint_tags import build_student_mlp_checkpoint_name, build_teacher_cache_name

# A100 optimisation: auto-tunes conv/matmul kernels after first batch
torch.backends.cudnn.benchmark = True


def load_teacher_cache(
    dataset_name,
    cache_tag=None,
    teacher_source="precompute",
    chronos2_cache_path=None,
    base_dir=None,
):
    """Try to load teacher outputs from precompute or Chronos2 cache."""
    project_root = base_dir if base_dir is not None else BASE_DIR

    if teacher_source == "chronos2":
        if chronos2_cache_path:
            cache_path = chronos2_cache_path
        else:
            cache_path = os.path.join(project_root, "results", "tsfm", f"cache_chronos2_{dataset_name}_c168_trainvaltest.pt")
        if not os.path.exists(cache_path):
            return None
        print(f"  Found Chronos2 cache: {cache_path}")
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        cache["source"] = "chronos2"
        return cache

    ds_info = DATASETS[dataset_name]
    cache_name = build_teacher_cache_name(dataset_name, cache_tag=cache_tag)
    cache_path = os.path.join(project_root, "data", ds_info["data_subdir"], cache_name)
    if os.path.exists(cache_path):
        print(f"  Found teacher cache: {cache_path}")
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        print(f"  Cached predictions shape: {cache['teacher_preds'].shape}")
        cache["source"] = "precompute"
        return cache
    return None


def train():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", action="store_true", help="Train student only (Baseline)")
    parser.add_argument("--dataset", type=str, default="metr-la",
                        choices=["metr-la", "pems-bay", "solar-al", "sdwpf", "acn-ev", "pems04", "pems03", "ecl", "gefcom-wind", "opsd-wind", "opsd-solar", "ercot"],
                        help="Dataset to train on")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable teacher cache (force live teacher inference)")
    parser.add_argument("--normalize", action="store_true",
                        help="Apply Z-score normalization (fitted on train split only)")
    parser.add_argument("--per-node-norm", action="store_true",
                        help="Use per-node Z-score (recommended for heterogeneous nodes)")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size (A100 default: 256)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--cache-tag", type=str, default=None,
                        help="Optional suffix for loading teacher cache filename")
    parser.add_argument("--teacher-source", type=str, default="precompute",
                        choices=["precompute", "chronos2"],
                        help="Teacher cache source")
    parser.add_argument("--chronos2-cache-path", type=str, default=None,
                        help="Explicit path to Chronos2 cache .pt file")
    parser.add_argument("--teacher-checkpoint", type=str, default=None,
                        help="Explicit teacher checkpoint filename for live KD mode")
    parser.add_argument("--checkpoint-tag", type=str, default=None,
                        help="Optional suffix for saved student checkpoint filename")
    parser.add_argument("--num-workers", type=int, default=8,
                        help="DataLoader num_workers (DGX A100 default: 8)")
    parser.add_argument("--loss", type=str, default="mse", choices=["mse", "mae", "huber"],
                        help="Loss function: mse, mae, or huber")
    parser.add_argument("--sensor-dropout", type=float, default=0.0,  # Paper 7
                        help="Max per-sample sensor-dropout rate (0 = off); see src/augment.py")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    # Seed
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 1. Configuration
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LR = 1e-3
    PATIENCE = 5
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # BF16 on A100: hardware-native, no loss-scaling needed
    use_amp = DEVICE == "cuda"
    amp_dtype = torch.bfloat16 if use_amp else torch.float32

    print(f"Using device: {DEVICE}")
    print(f"AMP dtype: {amp_dtype}")
    print(f"Arguments: {args}")

    # 2. Data Loading
    print("Loading Data...")
    try:
        data, edge_index, edge_weight, timestamps = load_traffic_data(args.dataset)
        print(f"Data Shape: {data.shape}")
    except Exception as e:
        print(f"Failed to load data: {e}")
        return

    if args.normalize:
        print(f"  Normalize: {args.normalize} (per-node={args.per_node_norm})")
        scaler = get_scaler(data, train_ratio=0.7, per_node=args.per_node_norm)
        data = scaler.transform(data)
    else:
        scaler = None

    # Split Train/Val/Test (70/10/20)
    L = len(data)
    train_size = int(L * 0.7)
    val_size = int(L * 0.1)

    train_data = data[:train_size]
    val_data = data[train_size:train_size+val_size]

    num_nodes = data.shape[1]
    edge_index = edge_index.to(DEVICE)

    # 3. Try loading teacher cache for KD mode
    use_cache = False
    teacher_cache = None
    teacher = None

    if not args.baseline and not args.no_cache:
        teacher_cache = load_teacher_cache(
            args.dataset,
            cache_tag=args.cache_tag,
            teacher_source=args.teacher_source,
            chronos2_cache_path=args.chronos2_cache_path,
        )
        if teacher_cache is not None:
            use_cache = True
            if teacher_cache.get("source") == "chronos2":
                cached_train_preds = teacher_cache["splits"]["train"]["forecast"].float()
                cached_val_preds = teacher_cache["splits"]["val"]["forecast"].float()
            else:
                train_dataset_tmp = SlidingWindowDataset(train_data, input_len=12, output_len=12)
                val_dataset_tmp = SlidingWindowDataset(val_data, input_len=12, output_len=12)

                n_train = len(train_dataset_tmp)
                n_val = len(val_dataset_tmp)

                cached_train_preds = teacher_cache["teacher_preds"][:n_train]
                cached_val_preds = teacher_cache["teacher_preds"][n_train:n_train+n_val]
            
            print(f"  Using CACHED teacher outputs (no live teacher needed!)")
            print(f"  Train cache: {cached_train_preds.shape}, Val cache: {cached_val_preds.shape}")

    # Create datasets
    train_dataset = SlidingWindowDataset(train_data, input_len=12, output_len=12)
    val_dataset = SlidingWindowDataset(val_data, input_len=12, output_len=12)

    if use_cache:
        # Wrap datasets to include cached teacher predictions
        # We need index-aware loading, so use custom collation
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        # We'll handle cache indexing via a different approach: 
        # Since shuffle=True breaks index mapping, pre-attach teacher preds to dataset
        
        # Build augmented TensorDatasets
        train_x = torch.stack([train_dataset[i][0] for i in range(len(train_dataset))])
        train_y = torch.stack([train_dataset[i][1] for i in range(len(train_dataset))])
        train_tp = cached_train_preds[:len(train_dataset)]
        train_aug = TensorDataset(train_x, train_y, train_tp)
        dl_kwargs = dict(num_workers=args.num_workers, pin_memory=True,
                         persistent_workers=(args.num_workers > 0),
                         prefetch_factor=4 if args.num_workers > 0 else None)
        train_loader = DataLoader(train_aug, batch_size=BATCH_SIZE, shuffle=True, **dl_kwargs)

        val_x = torch.stack([val_dataset[i][0] for i in range(len(val_dataset))])
        val_y = torch.stack([val_dataset[i][1] for i in range(len(val_dataset))])
        val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=BATCH_SIZE,
                                shuffle=False, **dl_kwargs)
    else:
        dl_kwargs = dict(num_workers=args.num_workers, pin_memory=True,
                         persistent_workers=(args.num_workers > 0),
                         prefetch_factor=4 if args.num_workers > 0 else None)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, **dl_kwargs)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, **dl_kwargs)
    
    print(f"Data Loaded. Nodes: {num_nodes}, Train Batches: {len(train_loader)}")
    
    # 4. Model Initialization
    print("Initializing Student...")
    student = LightweightST(
        num_nodes=num_nodes,
        in_steps=12,
        out_steps=12,
        input_dim=1,
        channels=32
    ).to(DEVICE)
    torch._dynamo.config.suppress_errors = True  # fallback to eager if Triton compile fails
    student = torch.compile(student)  # JIT-compile for A100 throughput
    print("  torch.compile: applied to student model")
    
    if not args.baseline and not use_cache:
        # Only load teacher if NO cache available (live inference mode)
        print("Initializing Teacher (live inference mode)...")
        teacher = STChronosTeacher(
            chronos_model_id="amazon/chronos-t5-small",
            num_nodes=num_nodes,
            prediction_length=12,
            context_length=12,
            use_spatial=True
        ).to(DEVICE)
        
        if args.teacher_checkpoint:
            checkpoint_path = os.path.join(BASE_DIR, "models", args.teacher_checkpoint)
        else:
            checkpoint_path = os.path.join(BASE_DIR, "models", f"teacher_{args.dataset}_tuned.pth")
        if os.path.exists(checkpoint_path):
            print(f"Loading Teacher from {checkpoint_path}...")
            ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
            ckpt = remap_legacy_teacher_keys(ckpt)
            teacher.load_state_dict(ckpt)
        else:
            print("WARNING: Teacher checkpoint not found! Using raw Chronos (Zero-shot).")
            
        teacher.eval()
    
    # 5. Training
    mode_str = "Baseline" if args.baseline else ("KD (cached)" if use_cache else "KD (live teacher)")
    print(f"\n=== Training Student: {mode_str} ===")
    student_opt = optim.Adam(student.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(student_opt, mode='min', factor=0.5, patience=2, verbose=True)
    
    loss_name = args.loss.lower()
    if loss_name == "mae":
        criterion = nn.L1Loss()
    elif loss_name == "huber":
        criterion = nn.HuberLoss(delta=1.0)
    else:
        criterion = nn.MSELoss()
    print(f"  Loss: {loss_name.upper()}")
    
    MAX_GRAD_NORM = 1.0
    
    alpha = 0.5  # Task Loss weight
    beta = 0.5   # Distillation weight
    if args.baseline:
        alpha = 1.0
        beta = 0.0
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_wts = copy.deepcopy(student.state_dict())
    
    for epoch in range(EPOCHS):
        # --- TRAIN ---
        student.train()
        train_loss = 0
        train_steps = 0
        
        for batch in train_loader:
            if use_cache:
                x, y, teacher_pred_cached = batch
                x, y = x.to(DEVICE), y.to(DEVICE)
                teacher_pred_cached = teacher_pred_cached.to(DEVICE)
            else:
                x, y = batch
                x, y = x.to(DEVICE), y.to(DEVICE)
            
            student_opt.zero_grad()
            if args.sensor_dropout > 0:  # Paper 7
                x = sensor_dropout(x, args.sensor_dropout, zero_fill_value(scaler))

            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                # Student Forward
                student_pred = student(x, edge_index)
                task_loss = criterion(student_pred, y)

                distill_loss = torch.tensor(0.0, device=DEVICE)
                if not args.baseline:
                    if use_cache:
                        distill_loss = criterion(student_pred, teacher_pred_cached)
                    else:
                        with torch.no_grad():
                            teacher_pred = teacher(x, edge_index)
                        distill_loss = criterion(student_pred, teacher_pred)

                loss = alpha * task_loss + beta * distill_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), MAX_GRAD_NORM)
            student_opt.step()
            
            train_loss += loss.item()
            train_steps += 1
            
            if train_steps % 50 == 0:
                print(f"Epoch {epoch+1} | Batch {train_steps}/{len(train_loader)} | Loss: {loss.item():.4f}")
            
        avg_train_loss = train_loss / train_steps
        
        # --- VALIDATION ---
        student.eval()
        val_loss = 0
        val_steps = 0
        all_preds = []
        all_trues = []
        
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            for batch in val_loader:
                x, y = batch[0].to(DEVICE, non_blocking=True), batch[1].to(DEVICE, non_blocking=True)
                pred = student(x, edge_index)
                loss = criterion(pred, y)
                val_loss += loss.item()
                val_steps += 1

                all_preds.append(pred.cpu())
                all_trues.append(y.cpu())
                
        avg_val_loss = val_loss / val_steps
        
        # Compute Metrics (inverse-transform if normalized)
        all_preds = torch.cat(all_preds, dim=0)
        all_trues = torch.cat(all_trues, dim=0)
        if scaler is not None:
            all_preds = scaler.inverse_transform(all_preds).float()
            all_trues = scaler.inverse_transform(all_trues).float()
        mae, rmse, mape, nmae = compute_metrics(all_preds, all_trues)
        scheduler.step(avg_val_loss)
        
        current_lr = student_opt.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | MAE: {mae:.4f} RMSE: {rmse:.4f} MAPE: {mape:.2f}% | LR: {current_lr:.6f}")
        
        # Checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_wts = copy.deepcopy(student.state_dict())
            patience_counter = 0
            ckpt_name = build_student_mlp_checkpoint_name(
                args.dataset,
                baseline=args.baseline,
                checkpoint_tag=args.checkpoint_tag,
            )
            torch.save(student.state_dict(), os.path.join(CKPT_DIR, ckpt_name))
        else:
            patience_counter += 1
            
        if patience_counter >= PATIENCE:
            print("Early Stopping.")
            break
            
    print("Training Complete.")
    print(f"Best Val Loss: {best_val_loss:.4f}")

    # Final test evaluation
    ckpt_name = build_student_mlp_checkpoint_name(
        args.dataset,
        baseline=args.baseline,
        checkpoint_tag=args.checkpoint_tag,
    )
    student.load_state_dict(torch.load(os.path.join(CKPT_DIR, ckpt_name),
                                       map_location=DEVICE, weights_only=True))
    student.eval()

    test_data = data[train_size + val_size:]
    test_dataset = SlidingWindowDataset(test_data, input_len=12, output_len=12)
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=4 if args.num_workers > 0 else None,
    )

    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            pred = student(x, edge_index)
            all_preds.append(pred.cpu())
            all_targets.append(y.cpu())

    all_preds = torch.cat(all_preds, dim=0).float().numpy()
    all_targets = torch.cat(all_targets, dim=0).float().numpy()

    if scaler is not None:
        all_preds = scaler.inverse_transform(all_preds)
        all_targets = scaler.inverse_transform(all_targets)

    mae = np.mean(np.abs(all_preds - all_targets))
    rmse = np.sqrt(np.mean((all_preds - all_targets) ** 2))
    mask = np.abs(all_targets) > 1.0
    mape = np.mean(np.abs((all_targets[mask] - all_preds[mask]) / (all_targets[mask] + 1e-6))) * 100 if mask.sum() > 0 else float("nan")
    nmae = mae / (np.mean(np.abs(all_targets)) + 1e-8)

    mode_label = "BASELINE" if args.baseline else "KD"
    print("=" * 60)
    print(f"FINAL TEST RESULTS (Student-MLP, {mode_label})")
    print("=" * 60)
    print(f"  Dataset: {args.dataset}")
    print(f"  Seed:    {args.seed}")
    print(f"  MAE:     {mae:.4f}")
    print(f"  RMSE:    {rmse:.4f}")
    print(f"  MAPE:    {mape:.2f}%")
    print(f"  nMAE:    {nmae:.4f}")
    print(f"  Saved:   {ckpt_name}")
    print("=" * 60)

if __name__ == "__main__":
    train()
