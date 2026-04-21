from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import List

from core.agent_runner import AgentRunner, build_default_multiturn_output_dir, run_agent_policy
from core.common import setup_logging, write_json
from core.dialogue_dataset import load_doc2dial_episodes
from core.eval_util import EvalConfig, build_eval_context, get_eval_log_path, load_train_config_from_checkpoint
from core.topology import build_nodes_and_edges


DEFAULT_EVAL_OUTPUTS_PATH = "outputs"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alg", default="mot")
    parser.add_argument("--checkpoint-dir-path", required=True)
    parser.add_argument("--split", default="train", choices=["train", "validation"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-episodes", type=int, default=30)
    parser.add_argument("--min-qa-turns", type=int, default=5)
    parser.add_argument("--max-turns-per-episode", type=int, default=10)
    parser.add_argument("--draft-max-new-tokens", type=int, default=48)
    parser.add_argument("--generation-max-new-tokens", type=int, default=64)
    parser.add_argument("--policy", default="all", choices=["all", "retain", "free"])
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--dialogue-json-path", default=None)
    parser.add_argument("--document-json-path", default=None)
    parser.add_argument("--doc2dial-root", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    output_dir = args.output_dir or build_default_multiturn_output_dir(args.checkpoint_dir_path)
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    eval_config = EvalConfig(
        alg=args.alg,
        outputs_path=DEFAULT_EVAL_OUTPUTS_PATH,
        timestamp=None,
        output_path=str(output_dir_path),
        checkpoint_dir_path=args.checkpoint_dir_path,
        device=args.device,
        batch_size=1,
        num_workers=0,
        max_examples_per_dataset=args.max_episodes,
        seed=args.seed,
        shuffle_eval_stream=False,
        shuffle_buffer=50000,
        generation_max_new_tokens=args.generation_max_new_tokens,
    )
    setup_logging(get_eval_log_path(eval_config.output_path))

    train_config = load_train_config_from_checkpoint(
        alg=args.alg,
        checkpoint_dir_path=args.checkpoint_dir_path,
        device_override=args.device,
    )
    nodes, edges = build_nodes_and_edges(train_config.model_ids, train_config.model_directions)
    ctx, translator_pool = build_eval_context(
        args.alg,
        eval_config,
        nodes,
        edges,
    )

    episodes = load_doc2dial_episodes(
        split=args.split,
        min_qa_turns=args.min_qa_turns,
        max_episodes=args.max_episodes,
        dialogue_json_path=args.dialogue_json_path,
        document_json_path=args.document_json_path,
        doc2dial_root=args.doc2dial_root,
    )
    if not episodes:
        raise RuntimeError("No Doc2Dial episodes were loaded. Check the split or dataset paths.")

    selected_policies: List[str]
    if args.policy == "all":
        selected_policies = ["retain", "free"]
    else:
        selected_policies = [args.policy]

    summaries = []
    for policy in selected_policies:
        debug_jsonl_path = output_dir_path / f"{policy}_debug.jsonl" if args.debug else None
        runner = AgentRunner(
            ctx=ctx,
            translator_pool=translator_pool,
            eval_config=eval_config,
            policy=policy,
            draft_max_new_tokens=args.draft_max_new_tokens,
            max_turns_per_episode=args.max_turns_per_episode,
            debug_enabled=args.debug,
            debug_output_path=str(debug_jsonl_path) if debug_jsonl_path is not None else None,
        )
        summary = run_agent_policy(
            runner=runner,
            episodes=episodes,
            output_dir=str(output_dir_path),
        )
        summaries.append(asdict(summary))
        print(
            f"[{policy}] episodes={summary.num_episodes} | completed={summary.num_completed_episodes} | "
            f"turns={summary.num_turns_completed} | avg_turn_f1={summary.average_turn_f1:.4f} | "
            f"avg_peak_live_kv_mib={summary.average_peak_live_kv_mib:.2f}"
        )

    summary_path = output_dir_path / "summary.json"
    write_json(str(summary_path), summaries)
    print(f"Saved summary to: {summary_path}")
    print(f"Saved run outputs to: {output_dir_path}")


if __name__ == "__main__":
    main()
