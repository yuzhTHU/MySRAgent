from __future__ import annotations

import numpy as np

from sr_agent.tools.constant_fit import ConstantFitTool


def test_requires_a_numeric_constant():
    x = np.linspace(1, 3, 40)
    tool = ConstantFitTool(data={"x": x, "y": 2 * x}, target="y")
    call = tool(eq="x")
    assert call.ok is True
    assert "no numerical constants" in call.result_str


def test_recognizes_fraction_and_pi_in_distinct_positions():
    x = np.linspace(1, 4, 80)
    validation_x = np.linspace(1.1, 3.9, 20)
    tool = ConstantFitTool(
        data={"x": x, "y": (2 / 3) * x + np.pi},
        evaluation_data={"x": validation_x, "y": (2 / 3) * validation_x + np.pi},
        target="y",
    )
    result = tool.execute(eq="0.6667*x + 3.1416")
    assert len(result["numbers"]) == 2
    assert "2/3" in [choice["label"] for choice in result["numbers"][0]["choices"]]
    assert "pi" in [choice["label"] for choice in result["numbers"][1]["choices"]]
    assert result["primary_metric"] == "validation_r2"
    assert result["pareto_front"][0]["constant_complexity"] == 0
    assert result["pareto_front"][0]["replacements"] == {"Number1": "2/3", "Number2": "pi"}
    assert result["pareto_front"][0]["validation_r2"] > 0.999999
    text = tool.format_result_dict(result)
    assert "    0.6667 -> ['2/3']" in text
    assert "    3.1416 -> ['pi', '22/7', 'sqrt(10)', '3']" in text
    assert "    (#Simplified Constants | Train-set R2 | Validation-set R2 | Formula)" in text
    assert "Pareto front (maximize #Simplified Constants and Validation-set R2):" in text
    assert "    2 | 1.00 | 1.00 | 2 / 3 * x + pi" in text
    assert "Number1=" not in text


def test_pareto_front_retains_accuracy_complexity_tradeoff():
    x = np.linspace(1, 4, 60)
    result = ConstantFitTool(data={"x": x, "y": 0.67 * x}, target="y").execute(eq="0.67*x")
    front = result["pareto_front"]
    assert [item["constant_complexity"] for item in front] == [0, 1]
    assert front[0]["formula"] == "2 / 3 * x"
    assert front[0]["train_r2"] < front[1]["train_r2"]
    assert front[1]["formula"] == "0.67 * x"
    text = ConstantFitTool.format_result_dict(result)
    assert "Pareto front (maximize #Simplified Constants and Train-set R2):" in text
    assert "0.999" in text
    assert "    0 | 1.00 | N/A | 0.67 * x" in text


def test_validation_r2_is_used_for_pareto_selection_when_available():
    x = np.linspace(1, 4, 60)
    xv = np.linspace(1.1, 3.9, 20)
    tool = ConstantFitTool(
        data={"x": x, "y": 0.67 * x},
        evaluation_data={"x": xv, "y": (2 / 3) * xv},
        target="y",
    )
    result = tool.execute(eq="0.67*x")
    assert [item["constant_complexity"] for item in result["pareto_front"]] == [0]
    assert result["pareto_front"][0]["train_r2"] < 1
    assert result["pareto_front"][0]["validation_r2"] == 1


def test_y_expression_takes_precedence_over_use_eq_as_y():
    x = np.linspace(1, 4, 50)
    data = {"x": x, "y": 0.5 * x + 1}
    tool = ConstantFitTool(data=data, target="y")
    result = tool.execute(eq="0.51*x", y="y-1", use_eq_as_y=True)
    assert result["target_expression"] == "y-1"
    assert result["primary_metric"] == "train_r2"
    assert any(item["formula"] == "1 / 2 * x" for item in result["pareto_front"])


def test_use_eq_as_y_compares_to_original_expression():
    x = np.linspace(1, 4, 50)
    tool = ConstantFitTool(data={"x": x, "y": x}, target="y")
    result = tool.execute(eq="0.67*x", use_eq_as_y=True)
    assert result["target_expression"] == "0.67 * x"
    assert result["pareto_front"][-1]["formula"] == "0.67 * x"
    assert result["pareto_front"][-1]["train_r2"] == 1.0


def test_repeated_numeric_value_is_treated_as_two_positions():
    x = np.linspace(1, 4, 50)
    tool = ConstantFitTool(data={"x": x, "y": 0.67 * x + 0.67}, target="y")
    result = tool.execute(eq="0.67*x+0.67")
    assert [item["id"] for item in result["numbers"]] == ["Number1", "Number2"]
    assert result["total_combinations"] == 4


def test_already_simple_one_is_omitted_from_nearby_display_but_still_counted():
    rng = np.random.default_rng(2026)
    x, z = rng.uniform(-1, 1, (2, 100))
    xv, zv = rng.uniform(-1.5, 1.5, (2, 30))
    tool = ConstantFitTool(
        data={"x": x, "z": z, "y": np.pi * x + np.sqrt(2) * z + 1},
        evaluation_data={"x": xv, "z": zv, "y": np.pi * xv + np.sqrt(2) * zv + 1},
        target="y",
    )
    result = tool.execute(eq="3.1416*x+1.4142*z+1")
    text = tool.format_result_dict(result)
    assert "    1 -> ['1']" not in text
    assert "Evaluated 50/50 combinations; invalid=0." in text
    assert "    3 | 1.00 | 1.00 | pi * x + sqrt(2) * z + 1" in text
