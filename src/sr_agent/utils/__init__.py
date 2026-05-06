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
from .render_markdown import render_markdown
from .factory_mixin import FactoryMixin
from .fix_parser import add_minus_flags, add_negation_flags
from .lazy_loader import setup_lazy_imports, TYPE_CHECKING

# 引入可选依赖的子模块
if TYPE_CHECKING:
    from .auto_gpu import AutoGPU
    from . import nn
__getattr__, __dir__, __all__ = setup_lazy_imports(__name__, {
    "AutoGPU": (".auto_gpu", "nn"), # 可选引入 .auto_gpu.AutoGPU, 但需要通过 pip install nd2py[nn] 来安装可选依赖
    "nn": (".nn", "nn"), # 将 .nn package 映射到 nn，并标明需要通过 pip install nd2py[nn] 来安装可选依赖
})
