from . import train, eval, info, report, capacity_curve


def register_subcommands(subparsers):
    train.add_subparser(subparsers)
    eval.add_subparser(subparsers)
    info.add_subparser(subparsers)
    report.add_subparser(subparsers)
    capacity_curve.add_subparser(subparsers)
