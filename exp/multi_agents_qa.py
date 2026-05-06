from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.agent_runner import AgentRunner, AgentRunnerConfig
from core.common import build_timestamp_string, setup_logging, write_json
from core.doc2dial_dataset import (
    DOC2DIAL_DEFAULT_DATA_DIR,
    DOC2DIAL_DEFAULT_URL,
    Doc2DialQAPair,
    load_doc2dial_qa_pairs,
)


def _str_to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y", "on"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run Doc2Dial multi-agent QA from official raw JSON files. "
            "Consecutive user turns are merged into one question, and the "
            "immediately following consecutive agent turns are merged into the gold answer."
        )
    )
    parser.add_argument("alg", choices=["mot", "interlat", "lsc", "c2c-pr", "kvcomm"], help="Algorithm to run.")
    parser.add_argument("--checkpoint-dir-path", required=True)
    parser.add_argument("--outputs-path", default="outputs/multiagent_scalability/Qwen-instruct")
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--device", default="auto")

    parser.add_argument(
        "--data-dir",
        default=DOC2DIAL_DEFAULT_DATA_DIR,
        help=(
            "Local Doc2Dial v1.0.1 directory. If this folder already exists and contains "
            "doc2dial_doc.json, doc2dial_dial_train.json, and doc2dial_dial_validation.json, "
            "it is reused. Otherwise the official zip is downloaded and extracted here."
        ),
    )
    parser.add_argument("--doc2dial-url", default=DOC2DIAL_DEFAULT_URL, help="Official Doc2Dial v1.0.1 zip URL.")
    parser.add_argument("--split", default="validation", help="Doc2Dial dialogue split, usually validation or train.")
    parser.add_argument("--domain", default=None, help="Optional Doc2Dial domain filter such as dmv, ssa, va, or uscis.")
    parser.add_argument(
        "--context-reference-roles",
        choices=["all", "user", "agent"],
        default="user",
        help=(
            "Which collapsed turns supply sp_id references for Base Context. "
            "user follows the requested setting: Base Context comes from the question/user turn sp_id; "
            "all also includes gold answer turns; agent uses only answer turns."
        ),
    )
    parser.add_argument(
        "--context-max-chars",
        type=int,
        default=None,
        help="Optional character truncation for Base Context after referenced text_sp spans are joined.",
    )

    parser.add_argument("--max-examples", type=int, default=10, help="Number of collapsed Doc2Dial QA examples to run.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle-eval-stream", nargs="?", const=True, default=False, type=_str_to_bool)
    parser.add_argument("--shuffle-buffer", type=int, default=1024)
    parser.add_argument("--max-turns", "--max-turn", dest="max_turns", type=int, default=4)
    parser.add_argument(
        "--agent-count",
        type=int,
        default=None,
        help=(
            "Number of agents to use from the checkpoint translator-pool nodes. "
            "Use 1 to evaluate the Hub-only single-turn baseline without communication. "
            "Default: use all available nodes, preserving the previous behavior."
        ),
    )
    parser.add_argument("--generation-max-new-tokens", type=int, default=48)
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


def _example_row(result, example: Doc2DialQAPair) -> Dict[str, Any]:
    return {
        "id": example.id,
        "dial_id": example.dial_id,
        "doc_id": example.doc_id,
        "domain": example.domain,
        "user_turn_ids": example.user_turn_ids,
        "agent_turn_ids": example.agent_turn_ids,
        "reference_sp_ids": example.reference_sp_ids,
        "missing_reference_sp_ids": example.missing_reference_sp_ids,
        "context": example.context,
        "question": result.question,
        "gold_answers": result.gold_answers,
        "prediction": result.prediction,
        "f1": result.f1,
        "peak_memory_bytes": result.profile.get("peak_memory_bytes"),
        "latency_sec": result.profile.get("latency_sec"),
        "agent_ids": result.agent_ids,
        "hub_agent_id": result.hub_agent_id,
        "cache_mode": result.cache_mode,
        "turns": [asdict(turn) for turn in result.turns],
        "transcript": result.transcript,
    }


def main() -> None:
    args = build_parser().parse_args()
    if args.alg in {"interlat", "lsc", "c2c-pr", "kvcomm"} and args.cache_mode != "retain":
        raise ValueError(f"alg={args.alg!r} supports only --cache-mode retain; free mode is not supported.")
    timestamp = build_timestamp_string()
    output_path = Path(args.output_path) if args.output_path else Path(args.outputs_path) / f"doc2dial_agent_runner_{args.alg}_{timestamp}"
    output_path.mkdir(parents=True, exist_ok=True)
    setup_logging(str(output_path / "eval.log"))

    examples = load_doc2dial_qa_pairs(
        data_dir=args.data_dir,
        url=args.doc2dial_url,
        split=args.split,
        max_examples=args.max_examples,
        shuffle=bool(args.shuffle_eval_stream),
        seed=args.seed,
        shuffle_buffer=args.shuffle_buffer,
        domain_filter=args.domain,
        context_reference_roles=args.context_reference_roles,
        context_max_chars=args.context_max_chars,
    )
    if not examples:
        raise RuntimeError(
            "No Doc2Dial QA examples were produced. Check --data-dir, --split, --domain, and turn references."
        )

    runner = AgentRunner.from_checkpoint(
        AgentRunnerConfig(
            alg=args.alg,
            checkpoint_dir_path=args.checkpoint_dir_path,
            device=args.device,
            max_turns=args.max_turns,
            generation_max_new_tokens=args.generation_max_new_tokens,
            max_prompt_tokens=args.max_prompt_tokens,
            agent_count=args.agent_count,
            seed=args.seed,
            log_turns=bool(args.log_turns),
            log_max_chars=args.log_max_chars,
            cache_mode=args.cache_mode,
        )
    )

    rows: List[Dict[str, Any]] = []
    total_f1 = 0.0
    peak_memory_bytes: Optional[int] = None

    for idx, example in enumerate(examples, start=1):
        result = runner.run(
            context=example.context,
            question=example.question,
            gold_answers=example.answers,
            example_index=idx,
        )
        total_f1 += float(result.f1)
        current_peak = result.profile.get("peak_memory_bytes")
        if current_peak is not None:
            peak_memory_bytes = int(current_peak) if peak_memory_bytes is None else max(peak_memory_bytes, int(current_peak))
        rows.append(_example_row(result, example))
        print(
            f"[{idx}/{len(examples)}] dial_id={example.dial_id} user_turns={example.user_turn_ids} "
            f"agent_turns={example.agent_turn_ids} F1={result.f1:.4f} | "
            f"prediction={result.prediction!r} | gold={result.gold_answers[:1]}"
        )

    count = len(rows)
    mean_f1 = total_f1 / count if count else float("nan")
    peak_memory_gib = float("nan") if peak_memory_bytes is None else peak_memory_bytes / (1024 ** 3)
    metrics = {
        "algorithm": args.alg,
        "cache_mode": args.cache_mode,
        "agent_count": len(runner.agent_sequence),
        "requested_agent_count": args.agent_count,
        "agent_ids": runner.node_ids,
        "hub_agent_id": runner.hub_agent.node_id,
        # "free_mode_peak_cache_agent_bound": runner._free_mode_peak_cache_agent_bound(),
        "checkpoint_dir_path": args.checkpoint_dir_path,
        "dataset": {
            "data_dir": args.data_dir,
            "doc2dial_url": args.doc2dial_url,
            "split": args.split,
            "domain": args.domain,
            "context_reference_roles": args.context_reference_roles,
            "context_max_chars": args.context_max_chars,
        },
        "count": count,
        "f1": mean_f1,
        "gpu_peak_memory_bytes": peak_memory_bytes,
        "gpu_peak_memory_gib": peak_memory_gib,
        "examples": rows,
        "args": vars(args),
    }
    metrics_path = output_path / "doc2dial_agent_runner_metrics.json"
    write_json(str(metrics_path), metrics)

    print("===== AgentRunner Doc2Dial sample =====")
    print(f"F1 Score: {mean_f1:.4f}")
    print(f"GPU Peak Memory: {peak_memory_gib:.3f} GiB")
    print(f"Saved metrics: {metrics_path}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
