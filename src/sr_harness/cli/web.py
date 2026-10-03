from __future__ import annotations

import argparse
from pathlib import Path


def setup_parser(parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    if parser is None:
        parser = argparse.ArgumentParser(
            prog="sr-harness web",
            description="Serve the SRHarness interactive workbench.",
        )
    else:
        parser.description = "Serve the SRHarness interactive workbench."
    parser.add_argument("--log-dir", default="logs", help=(
        "Directory containing SRHarness run logs."
    ))
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind.")
    parser.add_argument("--port", default=8000, type=int, help="Port to bind.")
    parser.add_argument("--reload", action="store_true", help="Enable uvicorn reload.")
    parser.add_argument("--no-browser", action="store_true", help=(
        "Do not open the workbench in a browser."
    ))
    parser.add_argument("--llm-provider", default="openrouter")
    parser.add_argument("--llm-model", default="deepseek/deepseek-v4-flash")
    return parser


def main(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit("Please install web dependencies with: pip install -e .[web]") from exc

    from sr_harness.web.app import create_app
    from sr_harness.runtime import InteractionController
    from sr_harness.web.session import InteractiveSession

    controller = InteractionController()
    agent_options = {
        "llm_provider": args.llm_provider,
        "llm_model": args.llm_model,
    }
    session = InteractiveSession(args.log_dir, controller, agent_options)
    app = create_app(Path(args.log_dir), controller=controller, session=session)
    browser_host = "127.0.0.1" if args.host in ("0.0.0.0", "::") else args.host
    url = f"http://{browser_host}:{args.port}"
    print(f"SRHarness Interactive: {url}", flush=True)
    if not args.no_browser:
        import threading
        import webbrowser
        timer = threading.Timer(1.5, webbrowser.open, args=(url,))
        timer.daemon = True
        timer.start()
    try:
        uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)
    finally:
        controller.command("stop")
        if session.thread:
            session.thread.join(timeout=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(setup_parser().parse_args()))
