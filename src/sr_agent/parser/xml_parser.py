# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
from __future__ import annotations
import re
import json
import inspect
from ast import literal_eval
from logging import getLogger
from typing import List, Dict, Any, Tuple
from .base_parser import BaseParser
from ..tools import BaseTool

_logger = getLogger(f'sr_agent.{__name__}')


@BaseParser.register('xml')
class XMLParser(BaseParser):
    pass