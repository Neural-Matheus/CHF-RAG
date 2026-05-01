import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_SRC_DIR = ROOT / "src"

if "src" not in sys.modules:
    _pkg = types.ModuleType("src")
    _pkg.__path__ = [str(_SRC_DIR)]
    sys.modules["src"] = _pkg


def _load(name: str):
    full = f"src.{name}"
    if full in sys.modules:
        return sys.modules[full]
    spec = importlib.util.spec_from_file_location(full, _SRC_DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full] = mod
    spec.loader.exec_module(mod)
    return mod



for _m in [
    "data_loader",
    "baselines",
    "ccg_rag",
    "coupled_heatflow",
    "heatflow_metrics",
    "question_typing",
    "hybrid_rerank",
]:
    _load(_m)
