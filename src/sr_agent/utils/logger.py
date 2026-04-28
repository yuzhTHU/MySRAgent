# Copyright (c) 2024-present, Yumeow. Licensed under the MIT License.
"""sr_agent 包的日志配置管理。

提供统一的日志配置接口，支持三种使用方式：

**方式 1：SRAgent 快速配置（最简单）**
    >>> from sr_agent import SRAgent
    >>> agent = SRAgent(..., verbose=True)  # 自动配置 DEBUG 级别日志
    >>> agent = SRAgent(..., verbose=True, save_path='logs/train.log')  # + 文件保存

**方式 2：手动调用 setup_logging()（可指定 exp_name）**
    >>> from sr_agent.utils import setup_logging
    >>> setup_logging(info_level='debug', exp_name='MyExp', save_path='logs/train.log')
    >>> agent = SRAgent(...)  # 使用已配置的日志

**方式 3：高级用户完全自定义**
    >>> import logging
    >>> logger = logging.getLogger('sr_agent')
    >>> logger.addHandler(my_custom_handler)  # 用户自己的 handler
    >>> setup_logging()  # 会检测到自定义 handler 并跳过

所有 sr_agent 子模块通过 logging.getLogger(f'sr_agent.{__name__}') 获取 logger，
自动从父 logger 'sr_agent' 继承 handlers。
"""
import re
import os
import sys
import time
import logging
from typing import Literal, Optional
from logging.handlers import RotatingFileHandler
from datetime import timedelta

__all__ = ["setup_logging", "LogFormatter", "is_logging_configured"]

# 全局状态：记录是否已配置 logging
_configured = False


def is_logging_configured() -> bool:
    """检查 sr_agent 的 logging 是否已配置。

    Returns:
        True 如果已配置过 logging（通过 setup_logging 或 SRAgent.__init__）
    """
    return _configured


class LogFormatter(logging.Formatter):
    """带颜色的日志格式化器。

    Attributes:
        exp_name: 实验名称，显示在日志前缀中。
        colorful: 是否启用彩色输出。
        start_time: 日志开始时间，用于计算相对时间。
    """

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
        self,
        exp_name: str = 'SR-Agent',
        colorful: bool = True,
        start_time: Optional[float] = None,
        time_format: str = "%b%d %H:%M:%S",
        show_lineno_for: tuple = ("TRACE", "WARNING", "ERROR", "CRITICAL"),
    ):
        super().__init__()
        self.exp_name = exp_name
        self.colorful = colorful
        self.start_time = start_time or time.time()
        self.time_format = time_format
        self.show_lineno_for = show_lineno_for

    def format(self, record) -> str:
        prefixes = [
            self.exp_name,
            record.name.split(".")[-1],  # 显示 logger 名的最后一部分
            record.levelname[0],  # D, I, N, W, E, C
            time.strftime(self.time_format),
            str(timedelta(seconds=record.created - self.start_time)),
        ]
        prefix = f"[{'|'.join([str(p) for p in prefixes if str(p).strip()])}]"

        if record.levelname in self.show_lineno_for:
            path = os.path.relpath(record.pathname, os.getcwd())
            prefix += f" ({path}:{record.lineno})"

        message = record.getMessage() or ""
        message = message.replace("\n", "\n" + " " * 8)  # 保持多行日志对齐

        if self.colorful:
            return (
                self.color_dict.get(record.levelname, "{}").format(prefix)
                + " "
                + message
            )
        else:
            return prefix + " " + re.sub(r"\033\[[\d;]+m", "", message)


# 自定义日志级别：NOTE (between INFO and WARNING) 和 TRACE (between DEBUG and INFO)
def note(self, message, *args, **kwargs):
    if self.isEnabledFor(25):
        self._log(25, message, args, **kwargs)


def trace(self, message, *args, **kwargs):
    if self.isEnabledFor(15):
        self._log(15, message, args, **kwargs)


def _setup_custom_levels():
    """注册自定义日志级别 NOTE 和 TRACE。"""
    if not hasattr(logging, 'NOTE'):
        logging.addLevelName(25, "NOTE")
        logging.Logger.note = note
        logging.NOTE = 25

    if not hasattr(logging, 'TRACE'):
        logging.addLevelName(15, "TRACE")
        logging.Logger.trace = trace
        logging.TRACE = 15


def _has_user_handlers() -> bool:
    """检查 sr_agent logger 是否已有用户自定义的 handler。

    通过检查 handler 类型来判断：
    - 库添加的 handler 会被标记（通过 _sr_agent_lib_handler 属性）
    - 用户添加的 handler 没有这些标记
    - NullHandler 是库默认添加的，不算用户自定义

    Returns:
        True 如果检测到用户自定义 handler
    """
    logger = logging.getLogger('sr_agent')
    for handler in logger.handlers:
        # NullHandler 是库默认添加的，不算用户自定义
        if isinstance(handler, logging.NullHandler):
            continue
        if not getattr(handler, '_sr_agent_lib_handler', False):
            return True
    return False


def setup_logging(
    info_level: Literal["debug", "trace", "info", "note", "warning", "error", "critical"] = "info",
    exp_name: str = 'SR-Agent',
    save_path: Optional[str] = None,
    colorful: bool = True,
    file_max_size_MB: float = 50.0,
    file_backup_count: int = 100,
    show_lineno_for_all_levels: bool = False,
    force: bool = False,
) -> bool:
    """配置 sr_agent 包的日志输出。

    这是配置 sr_agent 日志的主要入口。支持控制台彩色输出和文件保存。

    Args:
        info_level: 日志级别，默认 "info"。可选："debug", "trace", "info", "note", "warning", "error", "critical"。
        exp_name: 实验名称，显示在日志前缀中，默认 "SR-Agent"。
        save_path: 日志文件路径。None 表示不保存到文件。
        colorful: 是否启用控制台彩色输出，默认 True。
        file_max_size_MB: 单个日志文件最大大小（MB），默认 50。
        file_backup_count: 保留的备份文件数量，默认 100。
        show_lineno_for_all_levels: 是否对所有级别显示行号，默认 False。
        force: 是否强制重新配置（即使已有配置）。默认 False。

    Returns:
        True 如果成功配置，False 如果检测到用户自定义 handler 而跳过配置。

    Example:
        >>> from sr_agent.utils import setup_logging
        >>> setup_logging(info_level='debug', exp_name='MyExp', save_path='logs/train.log')
    """
    global _configured

    _setup_custom_levels()

    logger = logging.getLogger('sr_agent')

    # 检查是否有用户自定义 handler
    if not force and _has_user_handlers():
        logger.warning(
            "setup_logging detected user-configured handlers, skipping automatic configuration. "
            "Set force=True to override."
        )
        return False

    # 如果是 force 模式，清空已有 handlers
    if force:
        logger.handlers = []
    else:
        # 非 force 模式，移除 NullHandler（如果存在）
        logger.handlers = [h for h in logger.handlers if not isinstance(h, logging.NullHandler)]

    # 如果已配置且非 force 模式，直接返回
    if _configured and not force:
        return True

    # 设置 logger 基础配置
    logger.setLevel(logging.DEBUG)  # logger 本身放行所有级别
    logger.propagate = False

    # 准备 handler 级别
    show_lineno_for = ["TRACE", "WARNING", "ERROR", "CRITICAL"]
    if show_lineno_for_all_levels:
        show_lineno_for.extend(["DEBUG", "INFO", "NOTE"])

    # 记录 start_time 用于相对时间显示
    start_time = time.time()

    # ========== 控制台 Handler ==========
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.stream.reconfigure(encoding='utf-8')
    console_handler.setLevel(getattr(logging, info_level.upper()))
    console_handler.setFormatter(LogFormatter(
        exp_name=exp_name,
        colorful=colorful,
        start_time=start_time,
        show_lineno_for=show_lineno_for,
    ))
    console_handler._sr_agent_lib_handler = True  # 标记为库添加的 handler
    logger.addHandler(console_handler)

    # ========== 文件 Handler ==========
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        file_handler = RotatingFileHandler(
            save_path,
            mode="a",
            maxBytes=int(file_max_size_MB * 1024 * 1024),
            backupCount=file_backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)  # 文件记录所有级别
        file_handler.setFormatter(LogFormatter(
            exp_name=exp_name,
            colorful=False,  # 文件输出不带颜色
            start_time=start_time,
            show_lineno_for=show_lineno_for,
        ))
        file_handler._sr_agent_lib_handler = True  # 标记为库添加的 handler
        logger.addHandler(file_handler)

    _configured = True
    logger.debug(f"Logging configured: level={info_level}, exp_name={exp_name}, save_path={save_path}")

    return True


# 默认添加 NullHandler，避免 "No handler found" 警告
# NullHandler 是库添加的，不会阻止后续 setup_logging() 的自动配置
_root_logger = logging.getLogger('sr_agent')
_null_handler = logging.NullHandler()
_null_handler._sr_agent_lib_handler = True  # 标记为库添加的 handler
_root_logger.addHandler(_null_handler)
_root_logger.propagate = False
