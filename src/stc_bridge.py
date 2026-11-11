"""Single point of contact with the Paper 2 (ST-Chronos) codebase.

Paper 2's code is a READ-ONLY dependency. Nothing in Paper 7 may modify it.
Override the location with the STC_ROOT environment variable if needed.

Note: datasets_registry() and data_api() import Paper 2's `utils` package by name. Paper 7 must
therefore never create a top-level `utils/` or `models/` package; a regular package would shadow
Paper 2's the same way a Paper 7 `scripts/` package would have shadowed `scripts.evaluate_all`.
Keep Paper 7 modules under src/.
"""
import importlib.util
import os
import sys
from pathlib import Path

_DEFAULT = Path(__file__).resolve().parents[3] / "paper2_STC" / "STC-code"
_EVAL_MODULE = "stc_evaluate_all"


def stc_root() -> Path:
    return Path(os.environ.get("STC_ROOT", _DEFAULT)).resolve()


def _ensure_on_path() -> None:
    root = stc_root()
    if not root.is_dir():
        raise FileNotFoundError(
            f"Paper 2 (ST-Chronos) codebase not found at {root} (set STC_ROOT to override)"
        )
    root_s = str(root)
    if root_s not in sys.path:
        sys.path.insert(0, root_s)


def datasets_registry() -> dict:
    _ensure_on_path()
    from utils.data_loader import DATASETS
    return dict(DATASETS)


def data_api():
    """Return (load_traffic_data, SlidingWindowDataset, get_scaler)."""
    _ensure_on_path()
    from utils.data_loader import load_traffic_data, SlidingWindowDataset, get_scaler
    return load_traffic_data, SlidingWindowDataset, get_scaler


def model_loaders():
    """Return the checkpoint-loading helpers from Paper 2's evaluate_all.

    Loaded from an explicit file path, not `import scripts.evaluate_all`. Paper 7 has its
    own `scripts` package, and a regular package shadows Paper 2's namespace-portion
    `scripts/` no matter what order sys.path is in.
    """
    _ensure_on_path()
    path = stc_root() / "scripts" / "evaluate_all.py"
    if not path.is_file():
        raise FileNotFoundError(f"Paper 2 evaluate_all.py not found at {path}")
    if _EVAL_MODULE not in sys.modules:
        spec = importlib.util.spec_from_file_location(_EVAL_MODULE, path)
        module = importlib.util.module_from_spec(spec)
        # Registered before exec_module so the module survives its own internal imports.
        sys.modules[_EVAL_MODULE] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            del sys.modules[_EVAL_MODULE]
            raise
    module = sys.modules[_EVAL_MODULE]
    return module.load_dcrnn, module.load_student_mlp, module.load_student_attention


def norm_recommendation() -> dict:
    """Paper 2's per-dataset normalization policy ('global' or 'per_node')."""
    _ensure_on_path()
    from utils.data_loader import NORM_RECOMMENDATION
    return dict(NORM_RECOMMENDATION)
