from . import train, eval, info, confusion_matrix, report, capacity_curve


def register_subcommands(subparsers):
    train.add_subparser(subparsers)
    eval.add_subparser(subparsers)
    info.add_subparser(subparsers)
    confusion_matrix.add_subparser(subparsers)
    report.add_subparser(subparsers)
    capacity_curve.add_subparser(subparsers)
