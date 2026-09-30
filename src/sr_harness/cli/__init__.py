"""Unified command-line entry point for SRHarness."""
from __future__ import annotations
import argparse
import importlib

__all__ = ["entrypoint"]


COMMANDS = {
    "run": ("run", "Run SRAgent on a synthetic symbolic-regression problem."),
    "bench": ("bench", "Evaluate an algorithm with LLM-SRBench."),
    "tool": ("tool", "Inspect or invoke an SRHarness tool."),
    "web": ("web", "Serve the SRHarness search visualization."),
    "download-models": ("download_models", "Download a model checkpoint."),
    "upload-models": ("upload_models", "Upload a model checkpoint."),
}


class _CommandParser(argparse.ArgumentParser):
    def __init__(self, *args, module_name: str | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._module_name = module_name
        self._is_configured = module_name is None

    def parse_known_args(self, args=None, namespace=None):
        if not self._is_configured:
            self._is_configured = True
            module = importlib.import_module(f"{__package__}.{self._module_name}")
            module.setup_parser(self)
        return super().parse_known_args(args, namespace)


def setup_parser(parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    if parser is None:
        parser = argparse.ArgumentParser(prog="sr-harness", description="SRHarness command-line interface.")

    subparsers = parser.add_subparsers(
        dest="command",
        metavar="COMMAND",
        parser_class=_CommandParser,
    )
    for name, (module_name, help_text) in COMMANDS.items():
        subparsers.add_parser(name, help=help_text, module_name=module_name)
    return parser


def main(args: argparse.Namespace) -> int:
    module_name = COMMANDS[args.command][0]
    module = importlib.import_module(f"{__package__}.{module_name}")
    return module.main(args)


def entrypoint() -> int:
    parser = setup_parser()
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return 0
    return main(args)
