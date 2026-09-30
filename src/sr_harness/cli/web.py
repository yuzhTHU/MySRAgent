from __future__ import annotations

import argparse
from pathlib import Path


def setup_parser(
    parser: argparse.ArgumentParser | None = None,
) -> argparse.ArgumentParser:
    if parser is None:
        parser = argparse.ArgumentParser(
            prog="sr-harness web",
            description="Serve the SRHarness search visualization.",
        )
    else:
        parser.description = "Serve the SRHarness search visualization."
    parser.add_argument("--log-dir", default="logs", help="Directory containing SRHarness run logs.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind.")
    parser.add_argument("--port", default=8000, type=int, help="Port to bind.")
    parser.add_argument("--reload", action="store_true", help="Enable uvicorn reload.")
    return parser


def main(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit("Please install web dependencies with: pip install -e .[web]") from exc

    from sr_harness.web.app import create_app
    from sr_harness.web.interaction import InteractionController

    app = create_app(Path(args.log_dir), controller=InteractionController())
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(setup_parser().parse_args()))
