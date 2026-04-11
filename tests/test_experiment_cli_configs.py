from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
STUBS_PATH = REPO_ROOT / "tests" / "stubs"


def _build_env() -> dict[str, str]:
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(STUBS_PATH) + (os.pathsep + existing_pythonpath if existing_pythonpath else "")
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    return env


@pytest.mark.parametrize(
    "script_path",
    [
        "exp/layer_position.py",
        "exp/correction.py",
    ],
)
def test_experiment_scripts_use_default_config_for_target_layer_lookup(script_path: str) -> None:
    result = subprocess.run(
        [sys.executable, script_path, "--print-target-num-layers"],
        cwd=REPO_ROOT,
        env=_build_env(),
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "2"


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
    python_code = f"""
import json
from pathlib import Path

from common import build_dataclass_kwargs_from_json_and_namespace
from {module_name} import build_parser, {'LayerPositionConfig' if module_name.endswith('layer_position') else 'CorrectionConfig'}

config_cls = {'LayerPositionConfig' if module_name.endswith('layer_position') else 'CorrectionConfig'}
parser = build_parser()
args = parser.parse_args([
    '--default-config-path',
    str(Path({default_config_name!r})),
    *{extra_args!r},
])
kwargs = build_dataclass_kwargs_from_json_and_namespace(
    config_cls=config_cls,
    default_config_path=args.default_config_path,
    args=args,
    exclude_fields={{'alg'}},
)
print(json.dumps({{key: kwargs[key] for key in {list(expected)!r}}}, sort_keys=True))
"""
    result = subprocess.run(
        [sys.executable, "-c", python_code],
        cwd=REPO_ROOT,
        env=_build_env(),
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(result.stdout) == expected
