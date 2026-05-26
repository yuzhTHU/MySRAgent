# Copyright (c) 2024-present, Yumeow. Licensed under the MIT License.
from .attr_dict import *
from .logger import *
from .metrics import *
from .plot import *
from .timing import *
from .utils import *
from .tag2ansi import tag2ansi
from .log_exception import log_exception
from .classproperty import classproperty
from .render_python import render_python
from .symbolic_acc import get_symbolic_acc
from .render_markdown import render_markdown
from .factory_mixin import FactoryMixin
from .fix_parser import add_minus_flags, add_negation_flags
from .lazy_loader import setup_lazy_imports, TYPE_CHECKING
from .df_to_3line import df_to_3line
from .format_confusion_matrix import format_confusion_matrix
from .parse_json_with_template import parse_json_with_template
from .model_store import get_default, download_model, upload_model

# 引入可选依赖的子模块
if TYPE_CHECKING:
    from .auto_gpu import AutoGPU
    from . import nn
    from .load_model_state import load_model_state
__getattr__, __dir__, __all__ = setup_lazy_imports(__name__, {
    "AutoGPU": (".auto_gpu", "nn"), # 可选引入 .auto_gpu.AutoGPU, 但需要通过 pip install nd2py[nn] 来安装可选依赖
    "nn": (".nn", "nn"), # 将 .nn package 映射到 nn，并标明需要通过 pip install nd2py[nn] 来安装可选依赖
    "load_model_state": (".load_model_state", "nn"), # 引入 load_model_state 函数
})
