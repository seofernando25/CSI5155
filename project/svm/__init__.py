from . import benchmark, train, report

SVM_SUBCOMMANDS = {
    "extract-patches": "svm.extract_patches:main",
    "train-pca": "svm.train_pca:main",
    "transform-patches": "svm.transform_patches_pca:main",
    "compute-fv": "svm.compute_fisher_vectors:main",
    "eval": "svm.eval:main",
    "hparam": "svm.hparam:main",
}


def register_subcommands(subparsers, add_subcommand):
    for cmd, entry in SVM_SUBCOMMANDS.items():
        add_subcommand(subparsers, cmd, f"SVM {cmd.replace('-', ' ')}", entry)
    train.add_subparser(subparsers)
    benchmark.add_subparser(subparsers)
    report.add_subparser(subparsers)
