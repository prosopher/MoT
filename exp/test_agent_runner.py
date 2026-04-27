from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import torch

from core.agent_runner import AgentRunner, AgentRunnerConfig
from core.common import build_timestamp_string, setup_logging, write_json
from core.eval_util import (
    EvalConfig,
    build_generation_eval_dataloader,
    get_eval_log_path,
    get_squad_v11_dataset_spec,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a small SQuAD multi-agent KV-cache conversation experiment.")
    parser.add_argument("alg", choices=["c2c", "interlat", "lsc", "kvcomm", "mot", "mot-h"], help="Algorithm to run.")
    parser.add_argument("--checkpoint-dir-path", required=True)
    parser.add_argument("--outputs-path", default="outputs")
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-examples", type=int, default=16, help="Number of SQuAD examples to run; default is increased for a less tiny sample.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle-eval-stream", nargs="?", const=True, default=False, type=lambda value: str(value).lower() in {"1", "true", "yes", "y", "on"})
    parser.add_argument("--shuffle-buffer", type=int, default=1024)
    parser.add_argument("--max-turns", type=int, default=4)
    parser.add_argument("--generation-max-new-tokens", type=int, default=48)
    parser.add_argument("--max-prompt-tokens", type=int, default=None)
    parser.add_argument("--log-turns", dest="log_turns", action="store_true", default=True, help="Print every AgentRunner conversation turn to stdout/log file.")
    parser.add_argument("--no-log-turns", dest="log_turns", action="store_false", help="Disable per-turn AgentRunner logs.")
    parser.add_argument("--log-max-chars", type=int, default=600, help="Maximum prompt/response characters shown per turn log line.")
    parser.add_argument("--cache-mode", choices=["retain", "free"], default="retain", help="KV cache lifecycle mode: retain keeps all agent caches; free keeps the Hub cache and clears every non-hub cache after offload.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    timestamp = build_timestamp_string()
    output_path = Path(args.output_path) if args.output_path else Path(args.outputs_path) / f"agent_runner_{args.alg}_{timestamp}"
    output_path.mkdir(parents=True, exist_ok=True)

    eval_config = EvalConfig(
        alg=args.alg,
        outputs_path=args.outputs_path,
        timestamp=timestamp,
        output_path=str(output_path),
        checkpoint_dir_path=args.checkpoint_dir_path,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_examples_per_dataset=args.max_examples,
        seed=args.seed,
        shuffle_eval_stream=bool(args.shuffle_eval_stream),
        shuffle_buffer=args.shuffle_buffer,
        generation_max_new_tokens=args.generation_max_new_tokens,
    )
    setup_logging(get_eval_log_path(eval_config.output_path))

    runner = AgentRunner.from_checkpoint(
        AgentRunnerConfig(
            alg=args.alg,
            checkpoint_dir_path=args.checkpoint_dir_path,
            device=args.device,
            max_turns=args.max_turns,
            generation_max_new_tokens=args.generation_max_new_tokens,
            max_prompt_tokens=args.max_prompt_tokens,
            seed=args.seed,
            log_turns=bool(args.log_turns),
            log_max_chars=args.log_max_chars,
            cache_mode=args.cache_mode,
        )
    )

    spec = get_squad_v11_dataset_spec()
    dataloader = build_generation_eval_dataloader(spec=spec, eval_config=eval_config)

    rows: List[Dict[str, Any]] = []
    total_f1 = 0.0
    count = 0
    peak_memory_bytes = None

    for batch in dataloader:
        for example in batch:
            result = runner.run(
                context=example["context"],
                question=example["question"],
                gold_answers=example["answers"],
                example_index=count + 1,
            )
            count += 1
            total_f1 += float(result.f1)
            current_peak = result.profile.get("peak_memory_bytes")
            if current_peak is not None:
                peak_memory_bytes = int(current_peak) if peak_memory_bytes is None else max(peak_memory_bytes, int(current_peak))

            rows.append(
                {
                    "question": result.question,
                    "gold_answers": result.gold_answers,
                    "prediction": result.prediction,
                    "f1": result.f1,
                    "peak_memory_bytes": current_peak,
                    "agent_ids": result.agent_ids,
                    "hub_agent_id": result.hub_agent_id,
                    "cache_mode": result.cache_mode,
                    "turns": [asdict(turn) for turn in result.turns],
                    "transcript": result.transcript,
                }
            )
            print(
                f"[{count}/{args.max_examples}] F1={result.f1:.4f} | "
                f"prediction={result.prediction!r} | gold={result.gold_answers[:2]}"
            )
            if count >= args.max_examples:
                break
        if count >= args.max_examples:
            break

    mean_f1 = total_f1 / count if count else float("nan")
    peak_memory_gib = float("nan") if peak_memory_bytes is None else peak_memory_bytes / (1024 ** 3)
    metrics = {
        "algorithm": args.alg,
        "cache_mode": args.cache_mode,
        "agent_ids": runner.node_ids,
        "hub_agent_id": runner.hub_agent.node_id,
        "free_mode_peak_cache_agent_bound": runner._free_mode_peak_cache_agent_bound(),
        "checkpoint_dir_path": args.checkpoint_dir_path,
        "dataset": spec.name_for_log,
        "count": count,
        "f1": mean_f1,
        "gpu_peak_memory_gib": peak_memory_gib,
        "eval_config": asdict(eval_config),
        "examples": rows,
    }
    metrics_path = output_path / "agent_runner_metrics.json"
    write_json(str(metrics_path), metrics)

    print("===== AgentRunner SQuAD sample =====")
    print(f"F1 Score: {mean_f1:.4f}")
    print(f"GPU Peak Memory: {peak_memory_gib:.3f} GiB")
    print(f"Saved metrics: {metrics_path}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
