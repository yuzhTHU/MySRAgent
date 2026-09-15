import pytest

from sr_agent.utils.format_pareto_front import format_pareto_front


def candidate(formula="x + 1", validation_r2=0.98, train_r2=0.99, complexity=3):
    return {
        "formula": formula,
        "data_split_results": {
            "train": {"metrics": {"r2": train_r2, "complexity": complexity}},
            "validation": {"metrics": {"r2": validation_r2, "complexity": complexity}},
        },
        "node_id": "R1-C1-L1-K1",
    }


def test_concise_format_uses_nested_metrics_and_truncates_formula():
    output = format_pareto_front(
        [candidate(formula="x + y + z + 123456789")],
        concise=True,
        formula_max_length=12,
    )

    assert "minimize Complexity" in output
    assert "maximize Validation R²" in output
    assert "Complexity=3" in output
    assert "Validation R²=0.98" in output
    assert "Train R²=0.99" in output
    assert "Formula=x + y + z..." in output


def test_rich_format_uses_requested_columns_and_full_formula_by_default():
    formula = "x + y + z + 123456789"
    output = format_pareto_front(
        [candidate(formula=formula)],
        columns=["train.metrics.r2", "formula"],
        balance=(("train.metrics.complexity", "minimize"), ("train.metrics.r2", "maximize")),
    )

    assert "Train R²" in output
    assert "validation.metrics.r2" not in output
    assert formula in output
    assert "Balance: minimize train.metrics.complexity; maximize Train R²." in output


def test_rich_format_displays_floats_with_six_significant_digits():
    output = format_pareto_front([
        candidate(validation_r2=0.9980764883948635, train_r2=0.9981962354612538),
    ])

    assert "0.998076" in output
    assert "0.998196" in output
    assert "0.9980764883948635" not in output
    assert "0.9981962354612538" not in output


def test_empty_front_has_audience_specific_message():
    assert format_pareto_front([]) == "(empty)"
    assert "No Pareto front yet" in format_pareto_front([], concise=True)


def test_invalid_balance_and_formula_length_are_rejected():
    with pytest.raises(ValueError, match="exactly two"):
        format_pareto_front([candidate()], balance=(("train.metrics.r2", "maximize"),))
    with pytest.raises(ValueError, match="at least 4"):
        format_pareto_front([candidate(formula="a long formula")], formula_max_length=3)
