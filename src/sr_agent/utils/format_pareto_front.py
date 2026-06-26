import logging
import pandas as pd
from .df_to_3line import df_to_3line
from .log_exception import log_exception

_logger = logging.getLogger(f'sr_agent.{__name__}')


def format_pareto_front(pareto_front: list[dict]) -> str:
    """Format a Pareto front as a three-line terminal table."""
    if not pareto_front:
        return "(empty)"
    try:
        df = pd.DataFrame(pareto_front)[['mse', 'complexity', 'formula']]
        df.index = pd.Index(range(1, len(df) + 1), name="#")
        return df_to_3line(df)
    except Exception as e:
        _logger.error(f"Failed to format Pareto front table: {log_exception(e)}")
        return str(pareto_front)
