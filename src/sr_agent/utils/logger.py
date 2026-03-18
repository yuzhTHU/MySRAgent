# Copyright (c) 2024-present, Yumeow. Licensed under the MIT License.
import re
import os
import time
import logging
from typing import Literal
from logging.handlers import RotatingFileHandler
from datetime import timedelta

__all__ = ["init_logger"]


class LogFormatter(logging.Formatter):
    color_dict = {
        "DEBUG": "\033[0;37m{}\033[0m",
        "TRACE": "\033[1;448;5;240m{}\033[0m",
        "INFO": "\033[0;34m{}\033[0m",
        "NOTE": "\033[1;38;5;46m{}\033[0m",
        "WARNING": "\033[1;48;5;220m{}\033[0m",
        "ERROR": "\033[0;30;41m{}\033[0m",
        "CRITICAL": "\033[0;30;45m{}\033[0m",
    }

    def __init__(
        self, exp_name, colorful=False, start_time=None, time_format="%y-%m-%d %H:%M:%S",
        show_lineno_for=["TRACE", "WARNING", "ERROR", "CRITICAL"],
    ):
        super().__init__()
        self.exp_name = exp_name
        self.colorful = colorful
        self.start_time = start_time or time.time()
        self.time_format = time_format
        self.show_lineno_for = show_lineno_for

    def format(self, record):
        prefixes = [
            self.exp_name,
            record.name.split(".")[-1],
            record.levelname[0],  # D, I, N, W, E, C
            time.strftime(self.time_format),
            str(timedelta(seconds=record.created - self.start_time)),
        ]
        prefix = f"[{'|'.join([str(p) for p in prefixes if str(p).strip()])}]"
        if record.levelname in self.show_lineno_for:
            path = os.path.relpath(record.pathname, os.getcwd())
            prefix += f" ({path}:{record.lineno})"
        message = record.getMessage() or ""
        # message = message.replace("\n", "\n" + " " * len(prefix + " "))
        message = message.replace("\n", "\n" + " " * 8)
        if self.colorful:
            return (
                self.color_dict.get(record.levelname, "{}").format(prefix)
                + " "
                + message
            )
        else:
            return prefix + " " + re.sub(r"\033\[[\d;]+m", "", message)


# Logging level between INFO and WARNING, used for log something important but not unexpected.
def note(self, message, *args, **kwargs):
    if self.isEnabledFor(25):
        self._log(25, message, args, **kwargs)


# Logging level between DEBUG and INFO, used for warn something unexpected but not important.
def trace(self, message, *args, **kwargs):
    if self.isEnabledFor(15):
        self._log(15, message, args, **kwargs)


def init_logger(
    package_name: str,
    exp_name: str = None,
    log_file: str = None,
    info_level: Literal[
        "debug", "trace", "info", "note", "warning", "error", "critical"
    ] = "info",
    file_max_size_MB: float = 50.0,
    file_backup_count: int = 100,
    show_lineno_for_all_levels: bool = False,
):
    """Initialize the logger for the package.
    Args:
    - package_name: The name of the package / project, used in loggin.getLogger({package_name}.{path}.{to}.{file}).
    - exp_name: The name of the experiment, used in log prefix.
    - log_file: The path to the log file. If None, no file logging is performed.
    - info_level: The level of info logging. Can be one of 'debug', 'info', 'note', 'warning', 'error', 'critical'.
    - file_max_size_MB: The maximum size of the log file in MB. Default is 50MB.
    - file_backup_count: The number of backup files to keep. Default is 100.
    """
    start_time = time.time()

    logging.addLevelName(25, "NOTE")
    logging.Logger.note = note
    logging.NOTE = 25
    logging.addLevelName(15, "TRACE")
    logging.Logger.trace = trace
    logging.TRACE = 15

    logger = logging.getLogger(package_name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.handlers = []

    show_lineno_for = ["TRACE", "WARNING", "ERROR", "CRITICAL"]
    if show_lineno_for_all_levels:
        show_lineno_for.extend(["DEBUG", "INFO", "NOTE"])
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, info_level.upper()))
    console_handler.setFormatter(
        LogFormatter(
            exp_name, colorful=True, start_time=start_time, time_format="%b%d %H:%M:%S",
            show_lineno_for=show_lineno_for,
        )
    )
    logger.addHandler(console_handler)

    if log_file is not None:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = RotatingFileHandler(
            log_file,
            mode="a",
            maxBytes=int(file_max_size_MB * 1024 * 1024),
            backupCount=file_backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            LogFormatter(
                exp_name, colorful=False, start_time=start_time, time_format="%b%d %H:%M:%S",
                show_lineno_for=show_lineno_for,
            )
        )
        logger.addHandler(file_handler)


init_logger("nd2py")
