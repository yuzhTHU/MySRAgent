# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""TUI entry point for running SRAgentInteractive (powered by Textual)."""
from __future__ import annotations
import sys
import json
import time
import shlex
import logging
import asyncio
import argparse
from pathlib import Path
from textual import work
from inspect import signature
from datetime import datetime
from socket import gethostname
from sr_agent import SRAgentInteractive
from textual.containers import Vertical
from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, RichLog, Input, Static
from run_sr_agent import sanitize_filename, save_args, make_dataset
from sr_agent.utils import setup_logging, seed_all, log_exception, tag2ansi, LogFormatter


SCRIPT_NAME = Path(__file__).stem  # run_sr_agent_interactive_tui
_logger = logging.getLogger(f"sr_agent.{SCRIPT_NAME}")


def build_argparser() -> argparse.ArgumentParser:
    from run_sr_agent_interactive import build_argparser  # Reuse the same argparser as the non-TUI version
    parser = build_argparser()
    parser.description = "Run SRAgentInteractive on a symbolic-regression problem with human-in-the-loop (TUI)."
    parser.set_defaults(save_dir=f"./logs/{SCRIPT_NAME}", name=SCRIPT_NAME)
    parser.add_argument("--web", action="store_true", help="Serve the TUI as a web application on 0.0.0.0 (requires textual-serve).")
    parser.add_argument("--web_port", type=int, default=8080, help="Port for the web server (used with --web).")
    return parser


class TextualLogHandler(logging.Handler):
    """A logging handler that writes log records to a Textual RichLog widget."""

    def __init__(self, rich_log: RichLog):
        super().__init__()
        self.rich_log = rich_log

    def emit(self, record: logging.LogRecord):
        try:
            msg = self.format(record)
            # Use call_from_thread to safely post to the Textual event loop
            self.rich_log.app.call_from_thread(self._write_log, msg)
        except Exception:
            self.handleError(record)

    def _write_log(self, msg: str) -> None:
        """Write log with smart scroll: only auto-scroll if user is at the bottom."""
        rl = self.rich_log
        is_at_bottom = rl.scroll_offset.y >= (rl.virtual_size.height - rl.size.height - 2)
        rl.write(msg)
        if is_at_bottom:
            rl.scroll_end(animate=False)


class SRAgentTUI(App):
    """Textual TUI for SRAgentInteractive."""

    CSS = """
    #status-bar {
        height: 1;
        dock: top;
        background: $primary-background;
        color: $text;
        padding: 0 1;
        text-style: bold;
    }
    #log-panel {
        height: 1fr;
        border: solid green;
    }
    #input-box {
        dock: bottom;
        height: 3;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
    ]

    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.args = args
        self._pending_input_future: asyncio.Future | None = None
        self._agent: SRAgentInteractive | None = None
        self._start_time: float = time.time()

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static("Tokens: - | Cost: - | Time: - | Best: -", id="status-bar")
        with Vertical():
            yield RichLog(id="log-panel", highlight=True, markup=True, wrap=True, auto_scroll=False)
            yield Input(id="input-box", placeholder="Type your response here (Enter to submit)...")
        yield Footer()

    def _update_status_bar(self) -> None:
        """Periodically update the status bar with agent statistics."""
        elapsed = time.time() - self._start_time
        elapsed_str = ""
        if elapsed > 86400:  # more than a day
            elapsed_str += f"{int(elapsed // 86400)}day "
            elapsed %= 86400
        if elapsed > 3600:  # more than an hour
            elapsed_str += f"{int(elapsed // 3600)}h "
            elapsed %= 3600
        if elapsed > 60:  # more than a minute
            elapsed_str += f"{int(elapsed // 60)}m "
            elapsed %= 60
        elapsed_str += f"{int(elapsed % 60)}s"

        if self._agent is None:
            token_str = "-"
            money_str = "-"
        else:
            token = self._agent.token_counter.count
            money = self._agent.money_counter.count
            token_str = f"{token:,.0f}" if token else "0"
            money_str = f"${money:.4f}" if money else "$0"
        
        if self._agent is None:
            best_str = "Initializing..."
        elif not hasattr(self._agent, '_best_record'):
            best_str = "N/A"
        else:
            best_str = f"{self._agent._best_record['formula']} (MSE: {self._agent._best_record['mse']:.4f})"

        status = f"Tokens: {token_str} | Cost: {money_str} | Time: {elapsed_str} | Best: {best_str}"
        self.query_one("#status-bar", Static).update(status)

    def on_mount(self) -> None:
        """Set up the log handler and start the agent worker."""
        rich_log = self.query_one("#log-panel", RichLog)
        # Attach TextualLogHandler to the sr_agent root logger
        sr_logger = logging.getLogger("sr_agent")
        handler = TextualLogHandler(rich_log)
        handler.setLevel(logging.DEBUG if self.args.verbose else logging.INFO)
        console_handler = next((h for h in sr_logger.handlers if isinstance(h, logging.StreamHandler)), None)
        if console_handler and console_handler.formatter:
            exp_name = console_handler.formatter.exp_name
            self._start_time = start_time = console_handler.formatter.start_time
            show_lineno_for = console_handler.formatter.show_lineno_for
        else:
            exp_name = self.args.exp_name
            self._start_time = start_time = time.time()
            show_lineno_for = ["TRACE", "WARNING", "ERROR", "CRITICAL"]
        formatter = LogFormatter(
            exp_name=exp_name,
            start_time=start_time,
            colorful=False,
            show_lineno_for=show_lineno_for,
        )
        handler.setFormatter(formatter)
        sr_logger.addHandler(handler)
        # Start periodic status bar updates (every 2 seconds)
        self.set_interval(2.0, self._update_status_bar)
        self._run_agent()

    @work(thread=True)
    def _run_agent(self) -> None:
        """Run the agent in a background thread."""
        args = self.args
        features, target, formula, data = make_dataset(args)
        X = {name: data[name] for name in features}
        y = {target: data[target]}
        problem_description = args.problem_description or (
            f"Find the relationship {target} = f({', '.join(features)}). "
            f"The synthetic target was generated from an unknown formula."
        )
        _logger.note(
            f"Starting experiment {args.exp_name}\n"
            f"Equation: {target} = {formula}\n"
            f"Target variable: {target}; Feature variables: {', '.join(features)}\n"
            f"Generated {args.n_samples} samples with seed {args.seed}\n"
        )

        agent = SRAgentInteractive(
            llm_provider=args.llm_provider,
            llm_model=args.llm_model,
            tools=args.tools,
            local_sample_size=args.local_sample_size,
            max_refinement_depth=args.max_refinement_depth,
            global_width=args.global_width,
            max_restart_loop=args.max_restart_loop,
            restart_top_k=args.restart_top_k,
            verbose=args.verbose,
            tool_parser=args.tool_parser,
            save_path=args.save_path,
            workspace_files=args.workspace_files,
            use_workspace=args.use_workspace,
            max_workers=args.max_workers,
            human_input_callback=self._human_input_callback,
        )
        self._agent = agent

        # Wrap log_info to capture best formula for the status bar
        _log_info = agent.log_info
        def _patched_log_info(response_list, topk_records, R, L, C):
            if topk_records:
                agent._best_record = topk_records[0][-1]
            _log_info(response_list, topk_records, R=R, L=L, C=C)
        if signature(_log_info) == signature(agent.log_info):
            agent.log_info = _patched_log_info
            _logger.note("Patched agent.log_info to capture best formula for status bar.")
        else:
            _logger.warning("Signature of agent.log_info does not match expected. Skipping patch for best formula capture.")

        result = {
            "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "duration_seconds": None,
            "target_formula": f"{target} = {formula}",
            "noise_std_ratio": args.noise_std_ratio,
            "random_seed": args.seed,
            "best_formula": None,
            "best_mse": None,
            "status": "not_started",
            "progress": None,
            "token_usage": None,
            "money_usage": None,
            "tools_usage": None,
            "llm_model": f"{args.llm_model} @ {args.llm_provider}",
        }
        try:
            result |= agent.run(X=X, y=y, problem_description=problem_description)
        except KeyboardInterrupt as e:
            _logger.note("Experiment interrupted by user.")
            result |= getattr(e, "partial_result", {"status": "interrupted"})
        except Exception as e:
            _logger.error(f"Experiment failed with an exception: {log_exception(e)}")
            result |= getattr(e, "partial_result", {"status": "failed"})
            result["error"] = repr(e)
            if args.debug: raise
        finally:
            result["duration_seconds"] = (datetime.now() - datetime.strptime(result["start_time"], "%Y-%m-%d %H:%M:%S")).total_seconds()
            result["times_usage"] = agent.named_timer.to_str(mode='time', mode_of_detail='pace', mode_of_percent='by_time')
            result["token_usage"] = agent.token_counter.to_str(mode='count', mode_of_detail=None, mode_of_percent=None)
            result["money_usage"] = agent.money_counter.to_str(mode='count', mode_of_detail=None, mode_of_percent=None)
            result["tools_usage"] = agent.tools_counter.to_str(mode='count', mode_of_detail='count', mode_of_percent='by_count')
            # 打印日志
            truncate = lambda s: (s[:100] + '... [truncated]') if len(s) > 100 else s
            log = '\n'.join([f"[red]{k.replace('_', ' ').title()}[reset]: {truncate(str(v))}" for k, v in result.items()])
            _logger.note(tag2ansi(
                f'\n[gray]{"=" * 50}[reset]\n'
                "[red bold]Symbolic Regression Result[reset]\n"
                f"{log}\n"
                f'[gray]{"=" * 50}[reset]'
            ))
            # 保存文件
            result_path = Path(args.save_path) / "result.jsonl"
            with open(result_path, "a", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=True)
                f.write("\n")
            _logger.note(f"Result saved to {result_path}")

    def _human_input_callback(self, message: str) -> str:
        """Human input callback that uses the TUI Input widget.

        Called from the agent worker thread; blocks until user submits input.
        """
        self.call_from_thread(self._write_to_log, f"\n{'=' * 50}")
        self.call_from_thread(self._write_to_log, "[Agent asks for guidance]")
        self.call_from_thread(self._write_to_log, message)
        self.call_from_thread(self._write_to_log, f"{'=' * 50}")

        # Create a future on the main event loop and wait for it
        loop = self.app._loop  # noqa: SLF001
        future = asyncio.run_coroutine_threadsafe(self._wait_for_input(), loop)
        response = future.result()  # blocks the worker thread
        return response.strip() or "Continue with your best judgment."

    def _write_to_log(self, msg: str) -> None:
        """Write to log panel with smart scrolling."""
        rl = self.query_one("#log-panel", RichLog)
        rl.write(msg)
        if is_at_bottom := rl.scroll_offset.y >= (rl.virtual_size.height - rl.size.height - 2):
            rl.scroll_end(animate=False)

    async def _wait_for_input(self) -> str:
        """Async helper: sets up a future and waits for user to press Enter."""
        self._pending_input_future = asyncio.get_event_loop().create_future()
        input_widget = self.query_one("#input-box", Input)
        input_widget.focus()
        input_widget.placeholder = "Agent is waiting for your input..."
        result = await self._pending_input_future
        input_widget.placeholder = "Type your response here (Enter to submit)..."
        return result

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle Enter press on the input widget."""
        if self._pending_input_future and not self._pending_input_future.done():
            self._pending_input_future.set_result(event.value)
            event.input.value = ""


if __name__ == "__main__":
    parser = build_argparser()
    args, unknown = parser.parse_known_args()

    if args.exp_name is None:
        now = datetime.now()
        args.exp_name = sanitize_filename(
            f"{now:%Y%m%d}_{args.name}_{now:%H%M%S}_{gethostname()}"
        )
    else:
        args.exp_name = sanitize_filename(args.exp_name)
    if args.debug:
        args.verbose = True
    if args.seed == -1:
        args.seed = int(datetime.now().timestamp() * 1000) % (2**32 - 1)
    seed_all(args.seed)
    save_path = Path(args.save_dir) / args.exp_name
    save_path.mkdir(parents=True, exist_ok=True)
    args.save_path = str(save_path)
    args.command = " ".join(map(shlex.quote, [sys.executable, *sys.argv]))

    setup_logging(
        info_level="debug" if args.verbose else "info",
        exp_name=args.exp_name,
        save_path=save_path / "info.log",
        force=True,
    )

    if unknown:
        _logger.warning(f"Unknown args: {unknown}")
    _logger.note(f"Args: {args}")

    save_args(args, save_path / "args.json")

    if args.web:
        # Serve the TUI as a web app using textual-serve
        try:
            from textual_serve.server import Server
        except ImportError:
            raise SystemExit("ERROR: requires the 'textual-serve' package. Install it with: pip install textual-serve")
        # Rebuild command without --web to avoid infinite recursion
        cmd_args = [a for a in sys.argv if a != '--web']
        cmd = " ".join(map(shlex.quote, [sys.executable, *cmd_args]))
        host = "0.0.0.0"
        port = args.web_port
        _logger.note(f"Starting web server on http://{host}:{port}")
        server = Server(cmd, host=host, port=port)
        server.serve()
    else:
        app = SRAgentTUI(args)
        app.run()
    _logger.note(tag2ansi(f"Experiment completed. Re-run the script with [green bold]{args.command}[reset]"))
