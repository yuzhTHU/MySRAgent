# Copyright (c) 2024-present, Yumeow. Licensed under the MIT License.
import traceback


def log_exception(e: Exception) -> str:
    """Format exception for logging."""
    return f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
