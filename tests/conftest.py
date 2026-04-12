import importlib.util
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
STUBS_PATH = REPO_ROOT / "tests" / "stubs"

if str(STUBS_PATH) not in sys.path:
    sys.path.insert(0, str(STUBS_PATH))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(1, str(REPO_ROOT))


def _force_load_stub_package(package_name: str) -> None:
    package_dir = STUBS_PATH / package_name
    package_init_path = package_dir / "__init__.py"

    if not package_init_path.exists():
        raise FileNotFoundError(f"Missing stub package: {package_init_path}")

    for module_name in list(sys.modules):
        if module_name == package_name or module_name.startswith(f"{package_name}."):
            del sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(
        package_name,
        package_init_path,
        submodule_search_locations=[str(package_dir)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to build stub spec for {package_name}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = module
    spec.loader.exec_module(module)


_force_load_stub_package("transformers")
_force_load_stub_package("datasets")

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

try:
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass
