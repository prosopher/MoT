import argparse
import importlib

from core.eval_util import EvalConfig, build_eval_context, get_eval_log_path, resolve_latest_checkpoint_dir_for_alg
from core.common import add_dataclass_arguments, build_dataclass_kwargs_from_json_and_namespace, setup_logging


def load_eval_module(alg: str):
    module_name = f"alg.{alg}.eval"
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name:
            raise SystemExit(f"Unsupported alg: {alg}") from exc
        raise


def build_eval_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("alg")
    parser.add_argument(
        "--default-config-path",
        dest="default_config_path",
        default="configs/eval.json",
    )
    parser.add_argument("--dtype", default=None)
    add_dataclass_arguments(
        parser,
        EvalConfig,
        exclude_fields={"alg"},
    )
    return parser


def main() -> None:
    parser = build_eval_parser()
    args = parser.parse_args()

    eval_config_kwargs = build_dataclass_kwargs_from_json_and_namespace(
        config_cls=EvalConfig,
        default_config_path=args.default_config_path,
        args=args,
        exclude_fields={"alg"},
    )

    if eval_config_kwargs["checkpoint_dir_path"] is None:
        outputs_path = eval_config_kwargs["outputs_path"]
        eval_config_kwargs["checkpoint_dir_path"] = str(
            resolve_latest_checkpoint_dir_for_alg(args.alg, outputs_path=outputs_path)
        )

    eval_config = EvalConfig(
        alg=args.alg,
        **eval_config_kwargs,
    )

    setup_logging(get_eval_log_path(eval_config.output_path))

    eval_module = load_eval_module(args.alg)
    ctx, translator_pool, *extra = build_eval_context(
        args.alg,
        eval_config,
        dtype_override=args.dtype,
    )
    log_path = eval_module.run_eval(
        ctx,
        eval_config,
        translator_pool,
        *extra,
    )

    print(f"Evaluation log: {log_path}")


if __name__ == "__main__":
    main()
