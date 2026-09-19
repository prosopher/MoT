from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import statistics
import sys
from typing import Any, Dict, List

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.agent_runner import AgentRunner, AgentRunnerConfig, MALLM_SUPERMAJORITY_THRESHOLD
from core.common import setup_logging, write_json
from core.strategyqa_dataset import (
    STRATEGYQA_DEFAULT_DATA_DIR,
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
            "Run StrategyQA with the MALLM configuration: Expert personas, Memory discussion, "
            "Simple response generator, and Supermajority Consensus."
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
        help=(
            "Local StrategyQA cache directory. Missing task.json is downloaded from the same "
            "BIG-bench StrategyQA source used by MALLM."
        ),
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=10,
        help=(
            "Number of StrategyQA examples to run. When --start-example is greater than 1, this many examples are "
            "run starting from that 1-based evaluation-stream index unless --end-example is set."
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
        dest="max_turns",
        type=int,
        default=7,
        help=(
            "Maximum number of Agent turns. One verified Agent response is one turn; "
            "the first pass requires every Agent to participate once before consensus "
            "evaluation starts, then Supermajority Consensus (>66%) is evaluated after every turn."
        ),
    )
    parser.add_argument(
        "--agent-count",
        type=int,
        default=3,
        help=(
            "Number of logical Agents in the discussion. MALLM experiments use 3 agents by default; "
            "this remains independent of --max-turns."
        ),
    )
    parser.add_argument("--generation-max-new-tokens", type=int, default=1024)
    parser.add_argument(
        "--generation-temperature",
        type=float,
        default=1.0,
        help="Agent sampling temperature. Use 0 for greedy decoding.",
    )
    parser.add_argument("--max-prompt-tokens", type=int, default=None)
    parser.add_argument("--log-agents", dest="log_agents", action="store_true", default=True, help="Print per-agent AgentRunner logs.")
    parser.add_argument("--no-log-agents", dest="log_agents", action="store_false", help="Disable per-agent AgentRunner logs.")
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
        "question": example.question,
        "input": example.input_text,
        "choices": list(example.choices),
        "gold_answer": example.reference,
        "prediction": result.prediction,
        "accuracy": result.accuracy,
        "gpu_memory_gib": {
            "model_gib": result.profile.get("model_memory_gib"),
            "translator_gib": result.profile.get("translator_memory_gib"),
            "kv_gib": result.profile.get("kv_memory_gib"),
        },
        "latency_sec": result.profile.get("latency_sec"),
        "ttft_sec": result.profile.get("example_ttft_sec"),
        "agent_ids": result.agent_ids,
        "hub_agent_id": result.hub_agent_id,
        "personas": {
            agent_id: {"role": persona[0], "description": persona[1]}
            for agent_id, persona in result.personas.items()
        },
        "decision_protocol": "turn_supermajority_then_majority_vote",
        "consensus_requires_full_initial_participation": result.profile.get(
            "consensus_requires_full_initial_participation", True
        ),
        "supermajority_threshold": result.profile.get("supermajority_threshold", MALLM_SUPERMAJORITY_THRESHOLD),
        "supermajority_comparison": result.profile.get("supermajority_comparison", ">"),
        "consensus_reached": result.profile.get("consensus_reached"),
        "consensus_turn": result.profile.get("consensus_turn"),
        "final_decision_method": result.profile.get("final_decision_method"),
        "final_decision_answer": result.profile.get("final_decision_answer"),
        "turns": [asdict(turn_record) for turn_record in result.turns],
        "verification_retry_policy": result.profile.get("verification_retry_policy", "unbounded"),
        "verification_retry_count": result.profile.get("verification_retry_count", 0),
        "verification_failure_count": result.profile.get("verification_failure_count", 0),
        "cache_mode": result.cache_mode,
        "agent_messages": [asdict(message) for message in result.agent_messages],
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
        max_examples=load_max_examples,
        shuffle=bool(args.shuffle_eval_stream),
        seed=args.seed,
    )
    if not examples:
        raise RuntimeError("No StrategyQA examples were produced. Check --data-dir/task.json.")

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
            log_agents=bool(args.log_agents),
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
    memory_gib = {"model_gib": None, "translator_gib": None, "kv_gib": None}
    kv_cache_samples_gib: List[float] = []
    example_ttft_sec: List[float] = []

    selected_count = len(selected_examples)
    for local_idx, (example_index, example) in enumerate(selected_examples, start=1):
        try:
            result = runner.run(
                context="",
                question=example.input_text,
                gold_answers=example.answers,
                example_index=example_index,
            )
        except torch.cuda.OutOfMemoryError as error:
            oom_diagnostics: Dict[str, Any] = {
                "cache_mode": args.cache_mode,
                "agent_count": effective_agent_count,
                "example_index": example_index,
                "completed_examples": len(rows),
                "error": str(error),
            }
            write_json(str(output_path / "oom_diagnostics.json"), oom_diagnostics)
            raise
        total_accuracy += float(result.accuracy)
        kv_cache_samples_gib.extend(
            float(value) for value in result.profile.get("kv_cache_memory_samples_gib", [])
        )
        if result.profile.get("example_ttft_sec") is not None:
            example_ttft_sec.append(float(result.profile["example_ttft_sec"]))
        for component_key, profile_key in (
            ("model_gib", "model_memory_gib"),
            ("translator_gib", "translator_memory_gib"),
            ("kv_gib", "kv_memory_gib"),
        ):
            current_peak = result.profile.get(profile_key)
            if current_peak is None:
                continue
            previous_peak = memory_gib[component_key]
            memory_gib[component_key] = (
                float(current_peak) if previous_peak is None else max(float(previous_peak), float(current_peak))
            )
        rows.append(_example_row(result, example, example_index=example_index))
        print(
            f"[{local_idx}/{selected_count} | example={example_index}] "
            f"id={example.id} accuracy={result.accuracy:.0f} | "
            f"prediction={result.prediction!r} | gold={example.reference!r}\n"
        )

    count = len(rows)
    accuracy = total_accuracy / count if count else float("nan")
    memory_for_log = {
        key: float("nan") if value is None else float(value) for key, value in memory_gib.items()
    }
    metrics = {
        "algorithm": args.alg,
        "cache_mode": args.cache_mode,
        "discussion": "memory",
        "agent_count": len(runner.agent_sequence),
        "requested_agent_count": args.agent_count,
        "agent_ids": runner.node_ids,
        "hub_agent_id": runner.hub_agent.node_id,
        "persona_generator": "expert",
        "response_generator": "simple",
        "decision_protocol": "turn_supermajority_then_majority_vote",
        "consensus_requires_full_initial_participation": True,
        "supermajority_threshold": MALLM_SUPERMAJORITY_THRESHOLD,
        "supermajority_comparison": ">",
        "verification": {
            "retry_policy": "unbounded",
            "retry_count": sum(int(row.get("verification_retry_count", 0) or 0) for row in rows),
            "failure_count": sum(int(row.get("verification_failure_count", 0) or 0) for row in rows),
        },
        "checkpoint_dir_path": args.checkpoint_dir_path,
        "dataset": {
            "name": "StrategyQA",
            "data_dir": args.data_dir,
            "task_file": "task.json",
        },
        "example_range": {
            "start_example": args.start_example,
            "end_example": args.end_example,
            "max_examples": args.max_examples,
            "loaded_example_count": len(examples),
            "selected_example_ids": [example.id for _, example in selected_examples],
        },
        "count": count,
        "accuracy": accuracy,
        "ttft_sec": statistics.mean(example_ttft_sec) if example_ttft_sec else None,
        "gpu_memory_gib": {
            "model_gib": memory_gib["model_gib"],
            "translator_gib": memory_gib["translator_gib"],
            "kv_gib": memory_gib["kv_gib"],
        },
        "kv_cache_memory_stats_gib": {
            "sampling_basis": "stable_points_across_all_examples",
            "sample_count": len(kv_cache_samples_gib),
            "peak": max(kv_cache_samples_gib) if kv_cache_samples_gib else None,
            "mean": statistics.mean(kv_cache_samples_gib) if kv_cache_samples_gib else None,
            "median": statistics.median(kv_cache_samples_gib) if kv_cache_samples_gib else None,
        },
        "examples": rows,
        "args": vars(args),
    }
    metrics_path = output_path / "agent_runner_metrics.json"
    write_json(str(metrics_path), metrics)

    print("===== AgentRunner StrategyQA memory =====")
    print(f"Accuracy: {accuracy:.4f}")
    benchmark_ttft = metrics["ttft_sec"]
    print(
        "TTFT: "
        f"{benchmark_ttft if benchmark_ttft is not None else float('nan'):.4f} sec "
        "(Benchmark mean of Example mean Turn TTFTs; verification retries excluded)"
    )
    print(
        "GPU Memory: "
        f"model={memory_for_log['model_gib']:.3f} GiB | "
        f"translator={memory_for_log['translator_gib']:.3f} GiB | "
        f"kv_cache={memory_for_log['kv_gib']:.3f} GiB"
    )
    kv_stats = metrics["kv_cache_memory_stats_gib"]
    print(
        "KV Cache Memory: "
        f"peak={kv_stats['peak'] if kv_stats['peak'] is not None else float('nan'):.3f} GiB | "
        f"mean={kv_stats['mean'] if kv_stats['mean'] is not None else float('nan'):.3f} GiB | "
        f"median={kv_stats['median'] if kv_stats['median'] is not None else float('nan'):.3f} GiB"
    )
    print(f"Saved metrics: {metrics_path}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
