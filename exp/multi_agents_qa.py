from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys
from typing import Any, Dict, List

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.agent_runner import AgentRunner, AgentRunnerConfig
from core.common import setup_logging, write_json
from core.strategyqa_dataset import (
    STRATEGYQA_DEFAULT_DATA_DIR,
    STRATEGYQA_DEFAULT_SPLIT,
    StrategyQAExample,
    load_strategyqa_examples,
)


def _str_to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y", "on"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run StrategyQA with shared-memory multi-agent reasoning. "
            "Each Agent can use the full accumulated discussion; the final model acts as the Judge."
        )
    )
    parser.add_argument("alg", choices=["mot", "interlat", "lsc", "c2c-pr", "kvcomm"], help="Algorithm to run.")
    parser.add_argument("--checkpoint-dir-path", required=True)
    parser.add_argument(
        "--outputs-path",
        default="outputs/multi_agents",
        help="Root output directory. Default run directory: {algorithm}_{cache_mode}_{agent_count}.",
    )
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--device", default="auto")

    parser.add_argument(
        "--data-dir",
        default=STRATEGYQA_DEFAULT_DATA_DIR,
        help="Local StrategyQA directory. Missing train/dev JSON is downloaded from the official StrategyQA repository.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "dev"],
        default=STRATEGYQA_DEFAULT_SPLIT,
        help="StrategyQA split. Default: dev.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=10,
        help=(
            "Number of StrategyQA examples to run. When --start-example is greater than 1, this many examples are "
            "run starting from that 1-based global example index unless --end-example is set."
        ),
    )
    parser.add_argument(
        "--start-example",
        "--example-start",
        dest="start_example",
        type=int,
        default=1,
        help="1-based StrategyQA example index to start from after optional shuffling.",
    )
    parser.add_argument(
        "--end-example",
        "--example-end",
        dest="end_example",
        type=int,
        default=None,
        help="Optional 1-based inclusive StrategyQA example index to stop at.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle-eval-stream", nargs="?", const=True, default=False, type=_str_to_bool)
    parser.add_argument(
        "--max-turns",
        "--max-turn",
        dest="max_turns",
        type=int,
        default=4,
        help="Number of ordinary shared-memory discussion turns before one mandatory final Judge turn.",
    )
    parser.add_argument(
        "--agent-count",
        type=int,
        default=None,
        help=(
            "Number of logical Agents in the discussion. Use 1 for a single-model baseline; the final turn still uses "
            "the same model as Judge. Default: use all available nodes."
        ),
    )
    parser.add_argument("--generation-max-new-tokens", type=int, default=48)
    parser.add_argument(
        "--generation-temperature",
        type=float,
        default=1.0,
        help="Agent sampling temperature. MALLM experiments use temperature=1.0; use 0 for greedy decoding.",
    )
    parser.add_argument("--max-prompt-tokens", type=int, default=None)
    parser.add_argument("--log-turns", dest="log_turns", action="store_true", default=True, help="Print per-turn AgentRunner logs.")
    parser.add_argument("--no-log-turns", dest="log_turns", action="store_false", help="Disable per-turn AgentRunner logs.")
    parser.add_argument("--log-max-chars", type=int, default=600)
    parser.add_argument(
        "--cache-mode",
        choices=["retain", "free"],
        default="retain",
        help="KV cache lifecycle mode: retain keeps all agent caches; free keeps Hub cache and clears non-hub caches after offload.",
    )
    return parser


def _example_row(result, example: StrategyQAExample, *, example_index: int) -> Dict[str, Any]:
    return {
        "example_index": example_index,
        "id": example.id,
        "question": result.question,
        "gold_answer": example.answers[0],
        "prediction": result.prediction,
        "accuracy": result.accuracy,
        "facts": example.facts,
        "gpu_memory_gib": {
            "model_gib": result.profile.get("model_memory_gib"),
            "translator_gib": result.profile.get("translator_memory_gib"),
            "kv_gib": result.profile.get("kv_memory_gib"),
        },
        "latency_sec": result.profile.get("latency_sec"),
        "agent_ids": result.agent_ids,
        "judge_agent_id": result.hub_agent_id,
        "cache_mode": result.cache_mode,
        "turns": [asdict(turn) for turn in result.turns],
        "transcript": result.transcript,
    }


def main() -> None:
    args = build_parser().parse_args()
    if args.alg in {"interlat", "lsc", "c2c-pr", "kvcomm"} and args.cache_mode != "retain":
        raise ValueError(f"alg={args.alg!r} supports only --cache-mode retain; free mode is not supported.")
    if args.start_example < 1:
        raise ValueError(f"--start-example must be at least 1, got {args.start_example}")
    if args.end_example is not None and args.end_example < args.start_example:
        raise ValueError(
            f"--end-example must be greater than or equal to --start-example: "
            f"start={args.start_example}, end={args.end_example}"
        )
    if args.max_examples is not None and args.max_examples <= 0:
        raise ValueError(f"--max-examples must be positive when set, got {args.max_examples}")

    if args.end_example is not None:
        load_max_examples = args.end_example
    elif args.max_examples is not None:
        load_max_examples = args.start_example + args.max_examples - 1
    else:
        load_max_examples = None

    examples = load_strategyqa_examples(
        data_dir=args.data_dir,
        split=args.split,
        max_examples=load_max_examples,
        shuffle=bool(args.shuffle_eval_stream),
        seed=args.seed,
    )
    if not examples:
        raise RuntimeError("No StrategyQA examples were produced. Check --data-dir and --split.")

    indexed_examples = list(enumerate(examples, start=1))
    if args.end_example is not None:
        selected_examples = [
            (example_index, example)
            for example_index, example in indexed_examples
            if args.start_example <= example_index <= args.end_example
        ]
    else:
        selected_examples = indexed_examples[args.start_example - 1 :]
        if args.max_examples is not None:
            selected_examples = selected_examples[: args.max_examples]

    if not selected_examples:
        raise RuntimeError(
            f"No StrategyQA examples selected for start={args.start_example}, "
            f"end={args.end_example}, max_examples={args.max_examples}."
        )

    runner = AgentRunner.from_checkpoint(
        AgentRunnerConfig(
            alg=args.alg,
            checkpoint_dir_path=args.checkpoint_dir_path,
            device=args.device,
            max_turns=args.max_turns,
            generation_max_new_tokens=args.generation_max_new_tokens,
            generation_temperature=args.generation_temperature,
            max_prompt_tokens=args.max_prompt_tokens,
            agent_count=args.agent_count,
            seed=args.seed,
            log_turns=bool(args.log_turns),
            log_max_chars=args.log_max_chars,
            cache_mode=args.cache_mode,
        )
    )

    effective_agent_count = len(runner.agent_sequence)
    run_dir_name = f"{args.alg.replace('-', '_')}_{args.cache_mode}_{effective_agent_count}"
    output_path = Path(args.output_path) if args.output_path else Path(args.outputs_path) / run_dir_name
    output_path.mkdir(parents=True, exist_ok=True)
    setup_logging(str(output_path / "eval.log"))

    rows: List[Dict[str, Any]] = []
    total_accuracy = 0.0
    peak_memory_gib = {"model_gib": None, "translator_gib": None, "kv_gib": None}

    selected_count = len(selected_examples)
    for local_idx, (example_index, example) in enumerate(selected_examples, start=1):
        result = runner.run(
            context="",
            question=example.question,
            gold_answers=example.answers,
            example_index=example_index,
        )
        total_accuracy += float(result.accuracy)
        for component_key, profile_key in (
            ("model_gib", "model_memory_gib"),
            ("translator_gib", "translator_memory_gib"),
            ("kv_gib", "kv_memory_gib"),
        ):
            current_peak = result.profile.get(profile_key)
            if current_peak is None:
                continue
            previous_peak = peak_memory_gib[component_key]
            peak_memory_gib[component_key] = (
                float(current_peak) if previous_peak is None else max(float(previous_peak), float(current_peak))
            )
        rows.append(_example_row(result, example, example_index=example_index))
        print(
            f"[{local_idx}/{selected_count} | example={example_index}] "
            f"qid={example.id} accuracy={result.accuracy:.0f} | "
            f"prediction={result.prediction!r} | gold={example.answers[0]!r}"
        )

    count = len(rows)
    accuracy = total_accuracy / count if count else float("nan")
    peak_memory_total_gib = (
        float("nan")
        if any(value is None for value in peak_memory_gib.values())
        else sum(float(value) for value in peak_memory_gib.values() if value is not None)
    )
    peak_memory_for_log = {
        key: float("nan") if value is None else float(value) for key, value in peak_memory_gib.items()
    }
    metrics = {
        "algorithm": args.alg,
        "cache_mode": args.cache_mode,
        "discussion": "memory",
        "agent_count": len(runner.agent_sequence),
        "requested_agent_count": args.agent_count,
        "agent_ids": runner.node_ids,
        "judge_agent_id": runner.hub_agent.node_id,
        "checkpoint_dir_path": args.checkpoint_dir_path,
        "dataset": {
            "name": "StrategyQA",
            "data_dir": args.data_dir,
            "split": args.split,
        },
        "example_range": {
            "start_example": args.start_example,
            "end_example": args.end_example,
            "max_examples": args.max_examples,
            "loaded_example_count": len(examples),
            "selected_example_indices": [example_index for example_index, _ in selected_examples],
        },
        "count": count,
        "accuracy": accuracy,
        "gpu_peak_memory_gib": {
            "model_gib": peak_memory_gib["model_gib"],
            "translator_gib": peak_memory_gib["translator_gib"],
            "kv_gib": peak_memory_gib["kv_gib"],
        },
        "examples": rows,
        "args": vars(args),
    }
    metrics_path = output_path / "agent_runner_metrics.json"
    write_json(str(metrics_path), metrics)

    print("===== AgentRunner StrategyQA memory =====")
    print(f"Accuracy: {accuracy:.4f}")
    print(
        "GPU Peak Memory: "
        f"total={peak_memory_total_gib:.3f} GiB | "
        f"model={peak_memory_for_log['model_gib']:.3f} GiB | "
        f"translator={peak_memory_for_log['translator_gib']:.3f} GiB | "
        f"kv={peak_memory_for_log['kv_gib']:.3f} GiB"
    )
    print(f"Saved metrics: {metrics_path}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
