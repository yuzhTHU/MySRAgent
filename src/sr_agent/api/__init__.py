# Copyright (c) 2024-present, Yumeow. Licensed under the MIT License.
from .llm_api import LLMAPI, ToolList, ToolParserName
from .core import LLMResult
from .manual_api import ManualAPI
from .core import ToolCall
from ..utils import setup_lazy_imports, TYPE_CHECKING

# 引入可选依赖的子模块
if TYPE_CHECKING:
    from .openai_api import OpenAIAPI
    from .gemini_api import GeminiAPI
    from .deepseek_api import DeepSeekAPI
    from .openrouter_api import OpenRouterAPI
    from .siliconflow_api import SiliconFlowAPI
__getattr__, __dir__, __all__ = setup_lazy_imports(__name__, {
    # 可选引入 .openai.OpenAIPI, 但需要通过 pip install nd2py[all] 来安装可选依赖, 下同
    "OpenAIAPI": (".openai_api", "all"),
    "GeminiAPI": (".gemini_api", "all"),
    "DeepSeekAPI": (".deepseek_api", "all"),
    "OpenRouterAPI": (".openrouter_api", "all"),
    "SiliconFlowAPI": (".siliconflow_api", "all"),
})
