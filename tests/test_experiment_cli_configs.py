import importlib
import os
from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.common import build_dataclass_kwargs_from_json_and_namespace


STUBS_PATH = REPO_ROOT / "tests" / "stubs"


existing_pythonpath = os.environ.get("PYTHONPATH", "")
os.environ["PYTHONPATH"] = str(STUBS_PATH) + (os.pathsep + existing_pythonpath if existing_pythonpath else "")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


def _load_experiment_module(script_path: str):
    module_name = script_path.removesuffix(".py").replace("/", ".")
    module = importlib.import_module(module_name)
    config_cls_name = "LayerPositionConfig" if module_name.endswith("layer_position") else "CorrectionConfig"
    config_cls = getattr(module, config_cls_name)
    return module, config_cls


def _parse_config_kwargs(module, config_cls, cli_args: list[str]):
    parser = module.build_parser()
    args = parser.parse_args(cli_args)
    return build_dataclass_kwargs_from_json_and_namespace(
        config_cls=config_cls,
        default_config_path=args.default_config_path,
        args=args,
        exclude_fields={"alg"},
    )


@pytest.mark.parametrize(
    "script_path",
    [
        "exp/layer_position.py",
        "exp/correction.py",
    ],
)
def test_experiment_scripts_use_default_config_for_target_layer_lookup(script_path: str) -> None:
    module, config_cls = _load_experiment_module(script_path)
    config_kwargs = _parse_config_kwargs(module, config_cls, ["--print-target-num-layers"])

    if script_path == "exp/correction.py":
        resolve_target_num_layers = module.lp.resolve_target_num_layers
    else:
        resolve_target_num_layers = module.resolve_target_num_layers

    assert resolve_target_num_layers(config_kwargs["model_ids"], config_kwargs["model_directions"]) == 2


@pytest.mark.parametrize(
    ("module_name", "default_config_name", "extra_args", "expected"),
    [
        (
            "exp.layer_position",
            "configs/layer_position.json",
            [
                "--output-path",
                "override/layer_position",
                "--benchmark-mode",
                "logit_qa",
                "--eval-shuffle-stream",
                "--injection-layer-start-idx",
                "1",
            ],
            {
                "output_path": "override/layer_position",
                "benchmark_mode": "logit_qa",
                "eval_shuffle_stream": True,
                "injection_layer_start_idx": 1,
            },
        ),
        (
            "exp.correction",
            "configs/correction.json",
            [
                "--output-path",
                "override/correction",
                "--benchmark-mode",
                "logit_qa",
                "--injection-layer-start-idx",
                "1",
            ],
            {
                "output_path": "override/correction",
                "benchmark_mode": "logit_qa",
                "injection_layer_start_idx": 1,
            },
        ),
    ],
)
def test_experiment_cli_args_override_json_defaults(
    module_name: str,
    default_config_name: str,
    extra_args: list[str],
    expected: dict[str, object],
) -> None:
    module = importlib.import_module(module_name)
    config_cls_name = "LayerPositionConfig" if module_name.endswith("layer_position") else "CorrectionConfig"
    config_cls = getattr(module, config_cls_name)

    config_kwargs = _parse_config_kwargs(
        module,
        config_cls,
        ["--default-config-path", str(Path(default_config_name)), *extra_args],
    )

    assert {key: config_kwargs[key] for key in expected} == expected
