"""
Parser 用一种自定义的方式处理工具调用：
1. 在初始化时，Parser 接受一个工具列表，获得每个工具的名称、描述和参数信息。
2. Parser 提供一个方法，将所有工具的信息格式化成一个字符串，供 LLM 参考。其中包含样例的调用格式。
3. Parser 提供一个方法，接受 LLM 的输出字符串，解析出 (零或一或若干个) 工具调用的名称和参数，并返回一个结构化的结果。

Parser 的设计初衷在于允许不支持 tool 参数的 API 也能使用工具调用的功能。
"""
from .base_parser import BaseParser
from .text_parser import TextParser
from .json_parser import JSONParser
from .openai_parser import OpenAIParser # <- 为了一致起见定义了这个类，但是它实际用不到
