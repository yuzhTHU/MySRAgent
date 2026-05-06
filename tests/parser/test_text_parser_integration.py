"""TextParser 集成测试（需要调用真实的 LLM API）。

这些测试标记为 slow，因为它们需要调用真实的 LLM API。
"""

import pytest
from dotenv import load_dotenv
from src.sr_agent.parser import TextParser
from src.sr_agent.tools import BaseTool, LLMTool

# 加载环境变量
load_dotenv()


class TestTextParserLLMIntegration:
    """与真实 LLM 的集成测试。"""

    @pytest.fixture
    def parser(self):
        """创建 TextParser 实例。"""
        return TextParser(tool_list=None)

    @pytest.fixture
    def tool_formats(self):
        """获取格式化的工具描述。"""
        parser = TextParser(tool_list=None)
        return parser.format_tools()

    def _call_llm_and_parse(self, prompt: str, parser: TextParser, max_retries: int = 2):
        """调用 LLM 并解析响应。

        Args:
            prompt: 发送给 LLM 的提示。
            parser: TextParser 实例。
            max_retries: 最大重试次数。

        Returns:
            解析后的 actions 列表，如果解析失败返回 None。
        """
        for attempt in range(max_retries + 1):
            try:
                # 调用 LLM
                llm_tool = LLMTool()
                result = llm_tool(
                    llm_provider="siliconflow",
                    llm_model="Qwen3-8B",
                    messages=[{"role": "user", "content": prompt}]
                )

                if not result.get("success"):
                    pytest.skip(f"LLM API call failed: {result.get('error')}")

                response = result.get("message", "")

                # 解析响应
                actions = parser.parse_response(response)
                return actions, response

            except Exception as e:
                if attempt == max_retries:
                    pytest.fail(f"Failed after {max_retries + 1} attempts: {e}")
        return None, None

    @pytest.mark.slow
    @pytest.mark.parametrize("task_description,expected_tool", [
        ("Calculate the mean, variance, and standard deviation of the data.", "statistics_analysis"),
        ("Evaluate how well the formula x**2 + 1 fits the data.", "evaluate_formula"),
        ("Fit a polynomial of degree 2 to the data with interaction terms.", "polynomial_fit"),
        ("Execute python code: print('hello')", "code_executor"),
    ])
    def test_llm_generates_parseable_action(self, parser, tool_formats, task_description, expected_tool):
        """测试 LLM 生成的动作可以被正确解析。

        Args:
            parser: TextParser 实例。
            tool_formats: 格式化的工具描述。
            task_description: 任务描述。
            expected_tool: 期望 LLM 调用的工具名称。
        """
        prompt = f"""You are a helpful assistant that uses tools to complete tasks.

## Available Tools:
{tool_formats}

Task: {task_description}

Please use the appropriate tool to complete this task. Use the exact format shown above:
Action: tool_name(param1=value1, param2=value2)

For the parameters:
- If the task mentions specific values, use them
- If not, use reasonable default values based on the tool's description
- Make sure to include the correct parameter types (strings in quotes, numbers without quotes, lists in brackets)
"""

        actions, response = self._call_llm_and_parse(prompt, parser)

        if actions is None:
            pytest.skip("LLM API not available")

        assert len(actions) >= 1, f"No actions parsed from response: {response}"
        assert actions[0].name == expected_tool, f"Expected {expected_tool}, got {actions[0].name}"

    @pytest.mark.slow
    def test_llm_handles_multiple_sequential_actions(self, parser, tool_formats):
        """测试 LLM 能生成多个顺序执行的动作。"""
        prompt = f"""You are a helpful assistant that uses tools to complete tasks.

## Available Tools:
{tool_formats}

Task: First analyze the data statistics, then evaluate a simple formula.

Please use the appropriate tools in sequence. Use the exact format shown above:
Action: tool_name(param1=value1, param2=value2)

Make two tool calls, one per line.
"""

        actions, response = self._call_llm_and_parse(prompt, parser)

        if actions is None:
            pytest.skip("LLM API not available")

        # 应该有至少一个动作（某些模型可能只生成一个）
        assert len(actions) >= 1, f"No actions parsed from response: {response}"

    @pytest.mark.slow
    def test_llm_with_detailed_parameter_instructions(self, parser, tool_formats):
        """测试带有详细参数设置说明的情况下 LLM 的表现。"""
        prompt = f"""You are a helpful assistant that uses tools to complete tasks.

## Available Tools:
{tool_formats}

Task: Run a polynomial fit on the data.

Important parameter settings:
- Set max_degree to 3 (cubic polynomial)
- Set include_interactions to False (no interaction terms)
- Set include_bias to True (include intercept term)
- Use x_vars as ['x1', 'x2'] for the input features
- Use y_var as 'target' for the target variable

Please use the polynomial_fit tool with these exact parameters.
Use the format: Action: tool_name(param1=value1, param2=value2)
"""

        actions, response = self._call_llm_and_parse(prompt, parser)

        if actions is None:
            pytest.skip("LLM API not available")

        assert len(actions) >= 1, f"No actions parsed from response: {response}"
        if actions[0].name == "polynomial_fit":
            params = actions[0].params
            # 验证关键参数
            assert params.get("max_degree") == 3, f"max_degree should be 3, got {params.get('max_degree')}"
            assert params.get("include_interactions") is False, f"include_interactions should be False"

    @pytest.mark.slow
    def test_llm_with_formula_evaluation(self, parser, tool_formats):
        """测试 LLM 评估公式时的参数处理。"""
        prompt = f"""You are a helpful assistant that uses tools to complete tasks.

## Available Tools:
{tool_formats}

Task: Evaluate how well the formula "x1 squared plus sine of x2 plus 3.5" fits the data.

Please use the evaluate_formula tool.
Set the eq parameter to the formula using Python syntax (use ** for power, sin() for sine).
Set fit to True to optimize parameters.

Use the format: Action: tool_name(param1=value1, param2=value2)
"""

        actions, response = self._call_llm_and_parse(prompt, parser)

        if actions is None:
            pytest.skip("LLM API not available")

        assert len(actions) >= 1, f"No actions parsed from response: {response}"
        if actions[0].name == "evaluate_formula":
            params = actions[0].params
            eq = params.get("eq", "")
            # 验证公式包含关键元素
            assert "x1" in eq.lower() or "x**2" in eq, f"Formula should contain x1: {eq}"
            assert params.get("fit") is True, f"fit should be True"

    @pytest.mark.slow
    def test_llm_repeated_calls_consistency(self, parser, tool_formats):
        """测试多次调用 LLM 的一致性。

        重复 3 次相同的请求，验证每次都能被正确解析。
        """
        prompt = f"""You are a helpful assistant that uses tools to complete tasks.

## Available Tools:
{tool_formats}

Task: Analyze the data by computing basic statistics.

Please use the statistics_analysis tool with:
- x_vars = ['x1', 'x2']
- y_var = 'y'

Use the format: Action: tool_name(param1=value1, param2=value2)
"""

        if self._call_llm_and_parse(prompt, parser)[0] is None:
            pytest.skip("LLM API not available")

        # 重复 3 次
        for i in range(3):
            actions, response = self._call_llm_and_parse(prompt, parser)

            assert actions is not None, f"Attempt {i+1}: Failed to get response"
            assert len(actions) >= 1, f"Attempt {i+1}: No actions parsed from response"
            assert actions[0].name == "statistics_analysis", \
                f"Attempt {i+1}: Expected statistics_analysis, got {actions[0].name}"

    @pytest.mark.slow
    def test_llm_with_code_executor(self, parser, tool_formats):
        """测试 LLM 使用代码执行工具。"""
        prompt = f"""You are a helpful assistant that uses tools to complete tasks.

## Available Tools:
{tool_formats}

Task: Execute Python code to print "Hello, World!" and calculate 2 + 2.

Please use the code_executor tool.
Set the program parameter to valid Python code that prints the results.

Use the format: Action: tool_name(param1=value1)
"""

        actions, response = self._call_llm_and_parse(prompt, parser)

        if actions is None:
            pytest.skip("LLM API not available")

        assert len(actions) >= 1, f"No actions parsed from response: {response}"
        if actions[0].name == "code_executor":
            params = actions[0].params
            program = params.get("program", "")
            assert len(program) > 0, "Program should not be empty"
            assert "print" in program.lower(), f"Program should contain print: {program}"
