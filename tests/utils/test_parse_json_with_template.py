from sr_harness.utils import parse_json_with_template


def test_parse_json_with_template_extracts_fenced_json():
    template = {"equation_list": [str], "detail": str}
    text = """
    Some explanation.

    ```json
    {"equation_list": ["x1 + x2"], "detail": "ok"}
    ```
    """

    assert parse_json_with_template(text, template) == {
        "equation_list": ["x1 + x2"],
        "detail": "ok",
    }


def test_parse_json_with_template_accepts_yaml_like_object():
    template = {"score": float, "detail": [{"claim_id": str, "score": float, "detail": str}]}
    text = "{score: 0.5, detail: [{claim_id: c1, score: 1, detail: good}]}"

    assert parse_json_with_template(text, template) == {
        "score": 0.5,
        "detail": [{"claim_id": "c1", "score": 1.0, "detail": "good"}],
    }


def test_parse_json_with_template_coerces_percentage_scores():
    template = {"score": float, "detail": [{"claim_id": str, "score": float, "detail": str}]}
    text = """
    Analysis first.
    {"score": "75%", "detail": [{"claim_id": "c1", "score": "0.8", "detail": "fine"}]}
    """

    assert parse_json_with_template(text, template)["score"] == 0.75
