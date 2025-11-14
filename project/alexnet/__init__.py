from . import train, eval, info


def register_subcommands(subparsers):
    train.add_subparser(subparsers)
    eval.add_subparser(subparsers)
    info.add_subparser(subparsers)
