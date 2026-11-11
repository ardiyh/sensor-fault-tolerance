"""Write configs/models_<ds>.json: every model instance evaluated in the full grid.

Groups (one row per checkpoint):
  persistence, seasonal_naive        zero-parameter references
  gru_p2                             Paper 2 "DCRNN" checkpoint; diffusion inactive -> node-wise GRU
  dcrnn_fixed                        DCRNN with the diffusion step fixed, trained here (1 seed)
  mlp_task_p2, mlp_kd_p2             Paper 2 Student-MLP checkpoints (one each)
  attn_task_rev1, attn_kd_rev1       Student-Attn task-only / KD+AT, matched rev1 runs, seeds 0-4
  mlp_task_p7, mlp_aug_p7            Student-MLP retrained here without / with sensor dropout, seeds 0-4
  attn_task_p7, attn_aug_p7          Student-Attn retrained here without / with sensor dropout, seeds 0-4
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
P2 = "p2:"  # resolved against Paper 2's models/ directory


def entries(ds):
    e = [{"label": "persistence", "name": "persistence", "group": "persistence", "path": None},
         {"label": "seasonal_naive", "name": "seasonal_naive", "group": "seasonal_naive", "path": None},
         {"label": "gru_p2", "name": "dcrnn", "group": "gru_p2", "path": f"{P2}dcrnn_{ds}.pth"},
         {"label": "dcrnn_fixed", "name": "dcrnn_fixed", "group": "dcrnn_fixed",
          "path": f"checkpoints/dcrnn_fixed_{ds}_p7.pth"}]
    if ds == "metr-la":
        mlp_task, mlp_kd = f"{P2}student_baseline_{ds}.pth", f"{P2}best_student_{ds}.pth"
    else:  # not in Paper 2's local models/, restored from the Drive backup (MAE matches Paper 2)
        mlp_task = f"checkpoints/drive/backup/student_baseline_{ds}.pth"
        mlp_kd = f"checkpoints/drive/backup/best_student_{ds}.pth"
    e += [{"label": "mlp_task_p2", "name": "student_mlp_baseline", "group": "mlp_task_p2", "path": mlp_task},
          {"label": "mlp_kd_p2", "name": "student_mlp_kd", "group": "mlp_kd_p2", "path": mlp_kd}]
    for s in range(5):
        e += [
            {"label": f"attn_task_rev1_s{s}", "name": "student_attn_baseline", "group": "attn_task_rev1",
             "seed": s, "path": f"checkpoints/drive/rev1/student_attention_baseline_{ds}_rev1parity_seed{s}.pth"},
            {"label": f"attn_kd_rev1_s{s}", "name": "student_attn_kd_at", "group": "attn_kd_rev1", "seed": s,
             "path": f"checkpoints/drive/rev1/best_student_attention_{ds}_rev1batchF_matchedkdat_{ds}_seed{s}_v1.pth"},
            {"label": f"mlp_task_p7_s{s}", "name": "student_mlp_baseline", "group": "mlp_task_p7", "seed": s,
             "path": f"checkpoints/student_baseline_{ds}_p7s{s}.pth"},
            {"label": f"mlp_aug_p7_s{s}", "name": "student_mlp_baseline", "group": "mlp_aug_p7", "seed": s,
             "path": f"checkpoints/student_baseline_{ds}_p7augs{s}.pth"},
            {"label": f"attn_task_p7_s{s}", "name": "student_attn_baseline", "group": "attn_task_p7", "seed": s,
             "path": f"checkpoints/student_attention_baseline_{ds}_p7s{s}.pth"},
            {"label": f"attn_aug_p7_s{s}", "name": "student_attn_baseline", "group": "attn_aug_p7", "seed": s,
             "path": f"checkpoints/student_attention_baseline_{ds}_p7augs{s}.pth"},
        ]
    return e


def main():
    (ROOT / "configs").mkdir(exist_ok=True)
    for ds in ("metr-la", "pems-bay"):
        path = ROOT / "configs" / f"models_{ds}.json"
        path.write_text(json.dumps(entries(ds), indent=1))
        print(f"wrote {path} ({len(entries(ds))} models)")


if __name__ == "__main__":
    main()
