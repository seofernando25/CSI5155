from . import download, process, explorer


def register_subcommands(subparsers):
    download_parser = subparsers.add_parser("download", help="Download base datasets")
    download_parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download datasets even if they already exist",
    )
    download_parser.set_defaults(entry=lambda args: download.run(force=args.force))

    process_parser = subparsers.add_parser(
        "process", help="Process datasets (resize/gray)"
    )
    process_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing processed datasets if present",
    )
    process_parser.set_defaults(entry=lambda args: process.run(force=args.force))

    explorer_parser = subparsers.add_parser(
        "explorer", help="Launch dataset explorer server"
    )
    explorer_parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    explorer_parser.add_argument(
        "--port", type=int, default=8000, help="Port to bind to"
    )
    explorer_parser.set_defaults(
        entry=lambda args: explorer.run(host=args.host, port=args.port)
    )
