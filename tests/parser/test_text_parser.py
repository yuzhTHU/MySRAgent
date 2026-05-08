"""TextParser 单元测试。"""

import pytest
from src.sr_agent.parser import TextParser
from src.sr_agent.tools import BaseTool


class TestTextParserFormatTools:
    """测试 format_tools 方法。"""

    def test_format_tools_returns_string(self):
        """测试 format_tools 返回字符串。"""
        parser = TextParser(tool_list=None)
        result = parser.format_tools()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_format_tools_contains_expected_sections(self):
        """测试 format_tools 包含预期的章节。"""
        parser = TextParser(tool_list=None)
        result = parser.format_tools()

        assert "## Available Tools:" in result
        assert "## Output Format:" in result
        assert "Action:" in result
        assert "Description" in result
        assert "Signature" in result
        assert "DocString" in result

    def test_format_tools_contains_all_tools(self):
        """测试 format_tools 包含所有注册的工具。"""
        parser = TextParser(tool_list=None)
        result = parser.format_tools()

        # 检查所有工具名称出现在输出中
        tool_names = [
            "statistics_analysis",
            "evaluate_formula",
            "submit_formula",
            "polynomial_fit",
            "code_executor",
        ]
        for name in tool_names:
            assert name in result

    def test_format_tools_signature_includes_type_hints(self):
        """测试签名包含类型注解。"""
        parser = TextParser(tool_list=None)
        result = parser.format_tools()

        # 检查类型注解格式
        assert "Optional[" in result or "str" in result or "int" in result
        assert "x:" in result  # 参数名后跟冒号

    def test_format_tools_description_is_english(self):
        """测试描述是英文的。"""
        parser = TextParser(tool_list=None)
        result = parser.format_tools()

        # 简单的英文检查：不包含常见中文字符
        chinese_chars = ["计算", "评估", "调用", "拟合", "执行"]
        for char in chinese_chars:
            assert char not in result, f"Found Chinese text: {char}"

    def test_format_tools_docstring_not_placeholder(self):
        """测试 docstring 不是占位符。"""
        parser = TextParser(tool_list=None)
        result = parser.format_tools()

        # 检查 docstring 不是 `...`
        assert "`...`" not in result
        assert "DocString**: ..." not in result


class TestTextParserParseResponse:
    """测试 parse_response 方法。"""

    def test_parse_single_action_simple_params(self):
        """测试解析单个 Action，简单参数。"""
        parser = TextParser(tool_list=None)
        response = "Action: statistics_analysis(x_vars=['x1', 'x2'], y_var='y')"

        actions = parser.parse_response(response)

        assert len(actions) == 1
        assert actions[0].name == "statistics_analysis"
        assert actions[0].params == {"x_vars": ["x1", "x2"], "y_var": "y"}

    def test_parse_single_action_numeric_params(self):
        """测试解析单个 Action，数值参数。"""
        parser = TextParser(tool_list=None)
        response = "Action: polynomial_fit(max_degree=3, include_interactions=True)"

        actions = parser.parse_response(response)

        assert len(actions) == 1
        assert actions[0].name == "polynomial_fit"
        assert actions[0].params["max_degree"] == 3
        assert actions[0].params["include_interactions"] is True

    def test_parse_multiple_actions(self):
        """测试解析多个连续的 Action。"""
        parser = TextParser(tool_list=None)
        response = """
Here's my analysis:

Action: statistics_analysis(x_vars=['x1'], y_var='y')

Now let me evaluate:
Action: evaluate_formula(eq='x1**2 + 1', y_var='y')
        """

        actions = parser.parse_response(response)

        assert len(actions) == 2
        assert actions[0].name == "statistics_analysis"
        assert actions[1].name == "evaluate_formula"

    def test_parse_action_no_params(self):
        """测试解析没有参数的 Action。"""
        parser = TextParser(tool_list=None)
        response = "Action: statistics_analysis()"

        actions = parser.parse_response(response)

        assert len(actions) == 1
        assert actions[0].name == "statistics_analysis"
        assert actions[0].params == {}

    def test_parse_action_with_nested_list(self):
        """测试解析包含嵌套列表的参数。"""
        parser = TextParser(tool_list=None)
        response = "Action: polynomial_fit(interaction_blacklist=[('x1', 'x2'), ('x3', 'x4')])"

        actions = parser.parse_response(response)

        assert len(actions) == 1
        assert actions[0].name == "polynomial_fit"
        blacklist = actions[0].params.get("interaction_blacklist", [])
        assert len(blacklist) == 2

    def test_parse_action_with_float_values(self):
        """测试解析包含浮点数值的参数。"""
        parser = TextParser(tool_list=None)
        response = "Action: code_executor(program='print(3.14159)')"

        actions = parser.parse_response(response)

        assert len(actions) == 1
        assert actions[0].name == "code_executor"
        assert "3.14159" in actions[0].params.get("program", "")

    def test_parse_ignores_non_action_lines(self):
        """测试非 Action 行被忽略。"""
        parser = TextParser(tool_list=None)
        response = """
Let me think about this...
The best approach is to analyze first.
Action: statistics_analysis(x_vars=['x1'])
Some more text here.
"""

        actions = parser.parse_response(response)

        assert len(actions) == 1
        assert actions[0].name == "statistics_analysis"


class TestTextParserRobustness:
    """测试解析的鲁棒性。"""

    def test_parse_empty_response(self):
        """测试空响应。"""
        parser = TextParser(tool_list=None)
        response = ""

        actions = parser.parse_response(response)

        assert len(actions) == 0

    def test_parse_response_no_actions(self):
        """测试没有 Action 的响应。"""
        parser = TextParser(tool_list=None)
        response = """
This is just regular text.
No tool calls here.
"""

        actions = parser.parse_response(response)

        assert len(actions) == 0

    def test_parse_malformed_action_still_parses(self):
        """测试格式不太规范的 Action 仍能解析。"""
        parser = TextParser(tool_list=None)
        # 多余的空格
        response = "Action:   statistics_analysis(  x_vars=['x1']  )"

        actions = parser.parse_response(response)

        assert len(actions) == 1
        assert actions[0].name == "statistics_analysis"

    def test_parse_action_with_extra_whitespace(self):
        """测试带有额外空格的 Action。"""
        parser = TextParser(tool_list=None)
        response = """
    Action: statistics_analysis(x_vars=['x1'])
"""

        actions = parser.parse_response(response)

        assert len(actions) == 1

    def test_parse_action_variations_in_format(self):
        """测试不同格式的 Action。"""
        parser = TextParser(tool_list=None)

        # 函数名和括号之间有空格
        response1 = "Action: statistics_analysis (x_vars=['x1'])"
        actions1 = parser.parse_response(response1)
        assert len(actions1) == 1

        # 正常格式
        response2 = "Action: statistics_analysis(x_vars=['x1'])"
        actions2 = parser.parse_response(response2)
        assert len(actions2) == 1

    def test_parse_string_with_commas_inside_quotes(self):
        """测试引号内有逗号的字符串参数。"""
        parser = TextParser(tool_list=None)
        response = "Action: code_executor(program='a = [1, 2, 3]')"

        actions = parser.parse_response(response)

        assert len(actions) == 1
        assert "1, 2, 3" in actions[0].params.get("program", "")

    def test_parse_mixed_quote_styles(self):
        """测试混合引号风格。"""
        parser = TextParser(tool_list=None)
        response = '''Action: statistics_analysis(x_vars=["x1", "x2"], y_var='y')'''

        actions = parser.parse_response(response)

        assert len(actions) == 1
        assert actions[0].params.get("x_vars") == ["x1", "x2"]
        assert actions[0].params.get("y_var") == "y"

    def test_parse_case_insensitive_param_names(self):
        """测试参数名不区分大小写（解析时会转小写）。"""
        parser = TextParser(tool_list=None)
        response = "Action: statistics_analysis(X_VARS=['x1'], Y_VAR='target')"

        actions = parser.parse_response(response)

        assert len(actions) == 1
        # 参数名会被转换为小写
        assert "x_vars" in actions[0].params
        assert "y_var" in actions[0].params

    def test_parse_complex_expression(self):
        """测试复杂表达式。"""
        parser = TextParser(tool_list=None)
        response = "Action: evaluate_formula(eq='x1**2 + sin(x2) * 3.5', fit=True)"

        actions = parser.parse_response(response)

        assert len(actions) == 1
        assert actions[0].name == "evaluate_formula"
        assert "x1**2" in actions[0].params.get("eq", "")
        assert actions[0].params.get("fit") is True

    def test_parse_unterminated_string_gracefully(self):
        """测试未终止的字符串能优雅处理。"""
        parser = TextParser(tool_list=None)
        # 这是一个边界情况，解析器应该能处理
        response = "Action: code_executor(program='incomplete"

        actions = parser.parse_response(response)
        # 不应该抛出异常，可能会返回某种形式的结果
        assert isinstance(actions, list)


class TestTextParserIntegration:
    """TextParser 集成测试。"""

    def test_format_then_parse_roundtrip(self):
        """测试格式化后解析的往返。"""
        parser = TextParser(tool_list=None)

        # 获取格式化的工具描述
        formatted = parser.format_tools()

        # 模拟一个基于格式化输出的响应
        # 注意：不使用 "tool_name" 这个示例名称，避免和 format 中的示例混淆
        sim_response = f"""
Based on the available tools described above:

I will use the statistics tool:
Action: statistics_analysis(x_vars=['x1', 'x2'], y_var='y')
"""

        actions = parser.parse_response(sim_response)

        assert len(actions) == 1
        assert actions[0].name == "statistics_analysis"

    def test_full_workflow_simulation(self):
        """模拟完整的工作流程。"""
        parser = TextParser(tool_list=None)

        # 模拟多轮对话
        responses = [
            "Action: statistics_analysis(x_vars=['phi0', 'phi1'], y_var='y')",
            """
After analyzing the data, let me evaluate a formula:
Action: evaluate_formula(eq='phi0**2 + phi1', y_var='y', fit=False)
""",
            """
I need to fit a polynomial:
Action: polynomial_fit(x_vars=['x1', 'x2'], max_degree=2, include_interactions=True)
""",
        ]

        for response in responses:
            actions = parser.parse_response(response)
            assert len(actions) >= 1, f"Failed to parse: {response}"
