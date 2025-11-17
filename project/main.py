import argparse
import importlib
import sys
from typing import Callable, List, Optional


def _load_module_entry(module_path: str) -> Callable:
    module_name, _, attr = module_path.partition(":")
    module = importlib.import_module(module_name)
    if attr:
        return getattr(module, attr)
    if hasattr(module, "main"):
        return getattr(module, "main")
    if hasattr(module, "run"):
        return getattr(module, "run")
    raise AttributeError(
        f"{module_path} has no callable entry point (expected main or run)"
    )


def _invoke_module(module_path: str) -> None:
    func = _load_module_entry(module_path)
    func()


def _add_subcommand(
    subparsers,
    name: str,
    help_text: str,
    module_path: str,
    aliases: Optional[List[str]] = None,
):
    parser = subparsers.add_parser(name, help=help_text, aliases=aliases or [])
    parser.set_defaults(entry=lambda _args, path=module_path: _invoke_module(path))
    return parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified entry point for CIFAR-10 experiments"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # SVM pipeline commands
    from svm import register_subcommands as register_svm

    svm_parser = subparsers.add_parser("svm", help="SVM pipeline utilities")
    svm_sub = svm_parser.add_subparsers(dest="svm_command", required=True)
    register_svm(svm_sub, _add_subcommand)

    # ScaledCNN commands
    from scaledcnn import register_subcommands as register_scaledcnn

    scaledcnn_parser = subparsers.add_parser(
        "scaledcnn", help="ScaledCNN for double descent experiments"
    )
    scaledcnn_sub = scaledcnn_parser.add_subparsers(
        dest="scaledcnn_command", required=True
    )
    register_scaledcnn(scaledcnn_sub)

    from data import register_subcommands as register_data

    data_parser = subparsers.add_parser("data", help="Dataset preparation utilities")
    data_sub = data_parser.add_subparsers(dest="data_command", required=True)
    register_data(data_sub)

    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args, remainder = parser.parse_known_args(argv)

    entry = getattr(args, "entry", None)
    if entry is None:
        parser.print_help()
        sys.exit(1)

    if remainder:
        sys.argv = [sys.argv[0]] + remainder

    if entry is not None:
        entry(args)


if __name__ == "__main__":
    main()
