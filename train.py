import argparse
import importlib
from pathlib import Path

from core.common import add_dataclass_arguments, build_dataclass_kwargs_from_json_and_namespace, setup_logging
from core.context import Context
from core.train_util import get_train_log_path


def load_train_module(alg: str):
    module_name = f"alg.{alg}.train"
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name:
            raise SystemExit(f"Unsupported alg: {alg}") from exc
        raise


def build_train_parser(alg: str):
    train_module = load_train_module(alg)

    parser = argparse.ArgumentParser()
    parser.add_argument("alg")
    parser.add_argument(
        "--default-config-path",
        dest="default_config_path",
        default=f"configs/train_{alg}.json",
    )
    add_dataclass_arguments(
        parser,
        train_module.TrainConfig,
        exclude_fields={"alg"},
    )
    if hasattr(train_module, "ChannelProfiler"):
        parser.add_argument(
            "--channel-profile-config-path",
            dest="channel_profile_config_path",
            default="configs/channel_profile.json",
        )
    return parser, train_module


def main() -> None:
    bootstrap_parser = argparse.ArgumentParser(add_help=False)
    bootstrap_parser.add_argument("alg")
    bootstrap_args, _ = bootstrap_parser.parse_known_args()

    parser, train_module = build_train_parser(bootstrap_args.alg)
    args = parser.parse_args()

    config_kwargs = build_dataclass_kwargs_from_json_and_namespace(
        config_cls=train_module.TrainConfig,
        default_config_path=args.default_config_path,
        args=args,
        exclude_fields={"alg"},
    )
    config = train_module.TrainConfig(
        alg=args.alg,
        **config_kwargs,
    )

    setup_logging(get_train_log_path(config.output_path))

    ctx = Context(config)
    if hasattr(train_module, "ChannelProfiler") and train_module.uses_channel_alignment(getattr(config, "layer_alignment", "")):
        profile_config = train_module.load_channel_profile_config(Path(args.channel_profile_config_path))
        ctx.cp = train_module.ChannelProfiler(ctx, profile_config)

    final_checkpoint = Path(train_module.run_train(ctx))

    print(f"Saved outputs to {final_checkpoint.parent}")
    print(f"Final checkpoint: {final_checkpoint}")


if __name__ == "__main__":
    main()
