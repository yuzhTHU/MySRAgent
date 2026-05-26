# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
from __future__ import annotations
import re
import time
import json
import math
import logging
import warnings
import numpy as np
import nd2py as nd
from .tag2ansi import tag2ansi
from fractions import Fraction
from dataclasses import dataclass
from typing import Any, Mapping, Sequence
from itertools import combinations, product
from typing import Any, Dict, Tuple, Sequence

__all__ = [ "get_symbolic_acc" ]
_logger = logging.getLogger(f"sr_agent.{__name__}")


def llm_judge_equivalence(
    f_true: nd.Symbol,
    f_pred: nd.Symbol,
    ranges: Dict[str, Tuple[float, float]],
    llm_provider,
    llm_model,
    max_retry = 3,
    retry_timeout = 5,
):
    from ..api.llm_api import LLMAPI # utils 内的模块原则上不应该依赖于外部代码，加上这个防止循环依赖
    from .parse_json_with_template import parse_json_with_template

    messages = []
    messages.append({'role': 'system', 'content': (
        f"You are judging symbolic-regression formulas."
        f""
        f"Decide whether the predicted formula is equivalent to the ground-truth formula over the given variable ranges."
        f"Accept tiny numeric constant differences, for example 3.999999 may be treated as 4 and 0.00001 may be treated as 0."
        f"Also accept domain-specific equivalence over the provided ranges, such as abs(a) == a if the ranges imply a >= 0."
        f""
        f"Return a JSON in this schema:"
        f"{{'reason': 'brief analysis', 'equivalent': true/false}}"
    )})
    messages.append({'role': 'user', 'content': (
        f"Ground truth:"
        f"{f_true.to_str()}"
        f""
        f"Predicted:"
        f"{f_pred.to_str()}"
        f""
        f"Variable ranges:"
        f"{"\n".join(f"- {name}: [{lo}, {hi}]" for name, (lo, hi) in ranges.items())}"
    )})
    api = LLMAPI.create(llm_provider=llm_provider, llm_model=llm_model)
    for _ in range(max_retry):
        content = ""
        try:
            for content, _, _ in api(messages, n=1, max_tokens=1024, temperature=0.0):
                pass
            _logger.debug(f"Response: {content!r}")
            return parse_json_with_template(content, {'reason': str, 'equivalent': bool})
        except Exception as e:
            _logger.trace(f"Failed to parse LLM response: [{type(e).__name__}]{str(e)}. Response was: {content}")
            time.sleep(retry_timeout)
    else:
        _logger.warning(f"Failed to parse LLM response after {max_retry} attempts.")
        return {'equivalent': None, 'reason': f"Failed to parse LLM response after {max_retry} attempts."}


def my_nsimplify(
    vals: List[float] | float, 
    constants: Dict[str, float] = {}, 
    tolerance=1e-5, 
    max_denominator=1000, 
    relation_bound=4, 
    max_relation_terms=2
) -> List[str] | str:
    constants = {str(k): float(v) for k, v in constants.items() if math.isfinite(float(v)) and float(v) != 0}
    if is_single := isinstance(vals, (int, float)):
        vals = [vals]
    else:
        vals = list(vals)
    ans = []

    for x in vals:
        x, candidates = float(x), []
        if not math.isfinite(x):
            ans.append(str(x)); continue

        q = Fraction(x).limit_denominator(max_denominator)
        v = float(q)
        if abs(v - x) <= tolerance * max(1, abs(v), abs(x)):
            e = str(q.numerator) if q.denominator == 1 else f"{q.numerator}/{q.denominator}"
            candidates.append({"expr": e, "err": abs(v - x), "score": len(e) + math.log2(abs(q.numerator) + 1) + math.log2(q.denominator + 1)})

        for name, c in constants.items():
            if abs(c - x) <= tolerance * max(1, abs(c), abs(x)):
                candidates.append({"expr": name, "err": abs(c - x), "score": len(name) - 3})

            q = Fraction(x / c).limit_denominator(max_denominator)
            v = float(q) * c
            if abs(v - x) <= tolerance * max(1, abs(v), abs(x)):
                s, qabs = "-" if q < 0 else "", abs(q)
                n, d = qabs.numerator, qabs.denominator
                e = f"{s}{name}" if n == d == 1 else f"{s}{n}*{name}" if d == 1 else f"{s}{name}/{d}" if n == 1 else f"{s}{n}*{name}/{d}"
                candidates.append({"expr": e, "err": abs(v - x), "score": len(e) + math.log2(n + 1) + math.log2(d + 1)})

        names = list(constants)
        for k in range(1, min(max_relation_terms, len(names)) + 1):
            for subset in combinations(names, k):
                cs = [constants[n] for n in subset]
                for a0 in range(1, relation_bound + 1):
                    for coeffs in product(range(-relation_bound, relation_bound + 1), repeat=k + 1):
                        if all(a == 0 for a in coeffs[:-1]): continue
                        rel = a0 * x + sum(a * c for a, c in zip(coeffs[:-1], cs)) + coeffs[-1]
                        v = x - rel / a0
                        if abs(v - x) > tolerance * max(1, abs(v), abs(x)): continue

                        parts, fracs = [], []
                        for a, name in zip(coeffs[:-1], subset):
                            q = Fraction(-a, a0); fracs.append(q)
                            if q == 0: continue
                            s, qabs = 1 if q > 0 else -1, abs(q)
                            n, d = qabs.numerator, qabs.denominator
                            term = name if n == d == 1 else f"{n}*{name}" if d == 1 else f"{name}/{d}" if n == 1 else f"{n}*{name}/{d}"
                            parts.append((s, term))

                        q = Fraction(-coeffs[-1], a0); fracs.append(q)
                        if q:
                            s, qabs = 1 if q > 0 else -1, abs(q)
                            term = str(qabs.numerator) if qabs.denominator == 1 else f"{qabs.numerator}/{qabs.denominator}"
                            parts.append((s, term))

                        e = "0" if not parts else "".join((p if i == 0 and s > 0 else f"-{p}" if i == 0 else f" + {p}" if s > 0 else f" - {p}") for i, (s, p) in enumerate(parts))
                        candidates.append({"expr": e, "err": abs(v - x), "score": len(e) + 5 + sum(math.log2(abs(f.numerator) + 1) + math.log2(f.denominator + 1) for f in fracs)})

        ans.append(min(candidates, key=lambda c: (c["score"], c["err"]))["expr"] if candidates else repr(x))

    return ans[0] if is_single else ans


def get_symbolic_acc(
    f_true: nd.Symbol,
    f_pred: nd.Symbol,
    data: Dict[str, np.ndarray],
    atol = 1e-8,
    rtol = 1e-6,
    nsimplify_tolerance = 1e-5,
    llm_judge: bool = True,
    llm_provider: str = "deepseek",
    llm_model: str = "deepseek-v4-flash",
    wait_for_human: bool = False,
    return_details: bool = False,
) -> Dict[str, Any]:
    """Check whether two formulas are numerically equivalent on ``X``.

    The default policy is domain/numeric equivalence, not strict global symbolic
    proof. It first compares ``f_pred`` and ``f_true`` on ``X``. If that fails,
    it applies ``nsimplify`` to ``f_pred`` and compares again.
    """
    for var in (
        [var.name for var in f_true.iter_preorder() if isinstance(var, nd.Variable)] +
        [var.name for var in f_pred.iter_preorder() if isinstance(var, nd.Variable)]
    ):
        if var in data:
            pass
        elif var.lower() == 'pi':
            data[var] = math.pi
        elif var.lower() == 'e':
            data[var] = math.e
        else:
            raise ValueError(f"Variable '{var}' not found in data and is not recognized as a constant.")
    
    def _symbolic_acc(y_true, y_pred):
        if not np.any(is_finite := np.isfinite(y_true)):
            return {'equivalent': False, 'reason': 'y_true is all non-finite'}
        if not np.all(np.isfinite(y_pred[is_finite])):
            return {'equivalent': False, 'reason': 'y_pred has non-finite values where y_true is finite'}
        if np.allclose(y_true[is_finite], y_pred[is_finite], atol=atol, rtol=rtol):
            return {'equivalent': True, 'reason': 'y_pred is close to y_true within tolerances'}
        else:
            return {'equivalent': False, 'reason': 'y_pred is not close to y_true within tolerances'}
    
    if True:
        y_true = f_true.eval(data)
        y_pred = f_pred.eval(data)
        result = _symbolic_acc(y_true, y_pred)

    if not result['equivalent']:
        f_pred2 = f_pred.copy()
        constants = {'PI': math.pi, 'E': math.e, 'sqrt(2)': math.sqrt(2), 'sqrt(3)': math.sqrt(3), 'sqrt(5)': math.sqrt(5)}
        numbers = [num for num in f_pred2.iter_preorder() if isinstance(num, nd.Number)]
        values = my_nsimplify([num.value for num in numbers], tolerance=nsimplify_tolerance, constants=constants)
        for num, value in zip(numbers, values):
            value = nd.parse(value, {'PI': math.pi, 'E': math.e})
            f_pred2 = f_pred2.replace(num, value)
        _logger.debug(f"Applied nsimplify to f_pred. Original: {f_pred.to_str()}, Simplified: {f_pred2.to_str()}")
        y_pred2 = f_pred2.eval(data)
        result2 = _symbolic_acc(y_true, y_pred2)
        result2['reason'] = f"nsimplified f_pred to obtain y_pred. {result2['reason']}"
        if result2['equivalent']:
            f_pred = f_pred2
            y_pred = y_pred2
            result = result2

    if llm_judge:
        var_ranges = {name: (np.nanmin(values), np.nanmax(values)) for name, values in data.items()}
        result2 = llm_judge_equivalence(
            f_pred=f_pred,
            f_true=f_true,
            ranges=var_ranges,
            llm_provider=llm_provider,
            llm_model=llm_model,
        )
        if result['equivalent'] == result2['equivalent']:
            result['reason'] += f"; LLM judgement agrees: {result2['reason']}"
        elif wait_for_human:
            foo = lambda x: tag2ansi('[blue]NONE[reset]' if x is None else ('[green]EQUIVALENT[reset]' if x else '[red]NOT EQUIVALENT[reset]'))
            accept = input(
                f"Numeric equivalent is {foo(result['equivalent'])}, "
                f"while LLM judges the formulas {foo(result2['equivalent'])} since {result2['reason']}:\n"
                f"  f_true = {f_true.to_str()}\n"
                f"  f_pred = {f_pred.to_str()}\n"
                f"Accept LLM judgement? [y/N] "
            ).strip().lower() in {"y", "yes", "true", "1", ""}
            if accept:
                result['equivalent'] = result2['equivalent']
                result['reason'] += f"; accepted LLM judgement: {result2['equivalent']} ({result2['reason']})"
            else:
                result['reason'] += f"; rejected LLM judgement: {result2['equivalent']} ({result2['reason']})"
        else:
            foo = lambda x: tag2ansi('[blue]NONE[reset]' if x is None else ('[green]EQUIVALENT[reset]' if x else '[red]NOT EQUIVALENT[reset]'))
            accept = result2['equivalent'] is not None
            _logger.warning(
                f"Numeric equivalent is {foo(result['equivalent'])}, "
                f"while LLM judges the formulas {foo(result2['equivalent'])} since {result2['reason']}:\n"
                f"  f_true = {f_true.to_str()}\n"
                f"  f_pred = {f_pred.to_str()}\n"
                f"{"Accept" if accept else "Reject"} LLM judgement without human review."
            )
            if accept:
                result['equivalent'] = result2['equivalent']
                result['reason'] += f"; accepted LLM judgement: {result2['equivalent']} ({result2['reason']})"
            else:
                result['reason'] += f"; rejected LLM judgement: {result2['equivalent']} ({result2['reason']})"

    return result if return_details else result['equivalent']
