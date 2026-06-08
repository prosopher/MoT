import sys
from pathlib import Path

import pytest

import eval as eval_entry
import train as train_entry
import alg.mot.train as mot_train_module


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_PATH = REPO_ROOT / "tests" / "configs"


@pytest.mark.parametrize(
    ("alg", "train_config_name"),
    [
        ("lsc", "train_lsc_smoke.json"),
        ("mot", "train_mot_smoke.json"),
        ("c2c", "train_c2c_smoke.json"),
        ("kvcomm", "train_kvcomm_smoke.json"),
    ],
)
def test_train_and_eval_cli_smoke(alg: str, train_config_name: str, tmp_path: Path, capsys, monkeypatch) -> None:
    outputs_path = tmp_path / "outputs"
    outputs_path.mkdir(parents=True, exist_ok=True)

    timestamp = f"pytest_{alg}"
    train_config_path = CONFIGS_PATH / train_config_name
    eval_config_path = CONFIGS_PATH / "eval_smoke.json"
    channel_profile_config_path = CONFIGS_PATH / "channel_profile_smoke.json"

    output_path = outputs_path / f"{alg}_{timestamp}"
    checkpoint_dir_path = output_path
    checkpoint_path = checkpoint_dir_path / "translators"
    train_log_path = output_path / "train.log"

    train_argv = [
        "train.py",
        alg,
        "--default-config-path",
        str(train_config_path),
        "--output-path",
        str(output_path),
        "--timestamp",
        timestamp,
        "--device",
        "cpu",
    ]
    if alg != "kvcomm":
        train_argv.extend(["--max-steps", "1"])
    if alg == "mot":
        train_argv.extend(["--channel-profile-config-path", str(channel_profile_config_path)])

        def fake_resolve_channels(ctx):
            if ctx.cm.get_channels("A_to_B") or ctx.cm.get_channels("B_to_A"):
                return
            ctx.cm.add_channel("A_to_B", src_layer_idx=0, dst_layer_idx=0)
            ctx.cm.add_channel("B_to_A", src_layer_idx=0, dst_layer_idx=0)

        monkeypatch.setattr(mot_train_module, "resolve_channels", fake_resolve_channels)
    monkeypatch.setattr(sys, "argv", train_argv)
    train_entry.main()
    train_stdout = capsys.readouterr().out

    assert checkpoint_path.is_dir(), f"missing checkpoint for {alg}: {checkpoint_path}"
    assert train_log_path.exists(), f"missing train.log for {alg}: {train_log_path}"
    assert "Final checkpoint:" in train_stdout

    eval_log_path = output_path / "eval.log"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval.py",
            alg,
            "--default-config-path",
            str(eval_config_path),
            "--output-path",
            str(output_path),
            "--checkpoint-dir-path",
            str(checkpoint_dir_path),
            "--device",
            "cpu",
        ],
    )
    eval_entry.main()
    eval_stdout = capsys.readouterr().out

    assert eval_log_path.exists(), f"missing eval.log for {alg}: {eval_log_path}"
    assert "Evaluation log:" in eval_stdout

    train_log = train_log_path.read_text(encoding="utf-8")
    eval_log = eval_log_path.read_text(encoding="utf-8")

    if alg == "kvcomm":
        assert "Starting KVComm layer selection" in train_log
    else:
        assert "Starting training" in train_log
    assert "Starting evaluation" in eval_log
    assert "Preparing validation dataloader for OpenWebText/validation" in eval_log
    assert "[OpenWebText/validation]" in eval_log
    assert "FINAL MARKDOWN SUMMARY" in eval_log
    assert "Loss" in eval_log
