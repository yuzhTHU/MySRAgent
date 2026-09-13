# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""Fit a low-complexity polynomial plus oscillatory interaction.

The frequency is searched using training observations only. This is useful for
equations where a state variable modulates a periodic forcing term.
"""

from __future__ import annotations

from typing import Any, Dict

import nd2py as nd
import numpy as np
from scipy.optimize import minimize_scalar

from .base_tool import BaseTool, ToolMetadata


@BaseTool.register("harmonic_interaction_fit")
class HarmonicInteractionFitTool(BaseTool):
    metadata = ToolMetadata(name="harmonic_interaction_fit")

    def execute(
        self,
        carrier: str,
        oscillator: str,
        y: str = None,
        degree: int = 2,
        include_bias: bool = False,
        max_cycles: float = 20.0,
        grid_size: int = 320,
        show_diagnostics: bool = True,
    ) -> Dict[str, Any]:
        """Fit y = polynomial(carrier) + carrier * sin(frequency * oscillator).

        Args:
            carrier: State variable or expression that multiplies the sine term.
            oscillator: Variable or expression used as the sine argument.
            y: Target variable; defaults to the discovery target.
            degree: Polynomial degree in carrier, from 1 through 4.
            include_bias: Whether to fit an additive constant.
            max_cycles: Largest number of cycles across the observed oscillator span.
            grid_size: Number of initial frequency-grid points, from 64 through 2048.
            show_diagnostics: Include residual diagnostics in the returned evaluation.
        """
        target_name = (y or self.context["target"]).strip().strip('"').strip("'")
        data = self.context["data"]
        carrier_symbol = self.parse_formula(carrier)
        oscillator_symbol = self.parse_formula(oscillator)
        target_symbol = self.parse_formula(target_name)
        target = np.asarray(target_symbol.eval(data), dtype=float).reshape(-1)
        state = np.asarray(carrier_symbol.eval(data), dtype=float).reshape(-1)
        phase = np.asarray(oscillator_symbol.eval(data), dtype=float).reshape(-1)
        if not (target.shape == state.shape == phase.shape):
            raise ValueError("Target, carrier, and oscillator must have the same shape.")
        finite = np.isfinite(target) & np.isfinite(state) & np.isfinite(phase)
        degree = min(max(int(degree), 1), 4)
        grid_size = min(max(int(grid_size), 64), 2048)
        if np.count_nonzero(finite) < degree + 4:
            raise ValueError("Too few finite observations for harmonic interaction fit.")
        yy, xx, tt = target[finite], state[finite], phase[finite]
        span = float(np.ptp(tt))
        if span <= 0:
            raise ValueError("Oscillator must vary across training observations.")

        base_terms = [np.ones_like(xx)] if include_bias else []
        base_terms.extend(xx ** power for power in range(1, degree + 1))
        base = np.column_stack(base_terms)
        min_frequency = 2 * np.pi / (4 * span)
        max_frequency = 2 * np.pi * max(float(max_cycles), 0.5) / span
        frequencies = np.linspace(min_frequency, max_frequency, grid_size)

        def fit_at(frequency: float):
            design = np.column_stack((base, xx * np.sin(frequency * tt)))
            coefficients, _, rank, _ = np.linalg.lstsq(design, yy, rcond=None)
            if rank < design.shape[1]:
                return float("inf"), coefficients
            residual = yy - design @ coefficients
            return float(np.mean(residual * residual)), coefficients

        scores = np.asarray([fit_at(float(freq))[0] for freq in frequencies])
        if not np.any(np.isfinite(scores)):
            raise ValueError("Harmonic design matrix is rank deficient at every frequency.")
        # Refining several grid minima avoids choosing a sidelobe of the sine fit.
        local_minima = [
            i for i in range(1, grid_size - 1)
            if scores[i] <= scores[i - 1] and scores[i] <= scores[i + 1]
        ]
        local_minima.extend((0, grid_size - 1, int(np.nanargmin(scores))))
        candidate_indices = sorted(set(local_minima), key=lambda i: scores[i])[:8]
        best = None
        for i in candidate_indices:
            left = float(frequencies[max(i - 1, 0)])
            right = float(frequencies[min(i + 1, grid_size - 1)])
            if left == right:
                continue
            refined = minimize_scalar(
                lambda frequency: fit_at(float(frequency))[0],
                bounds=(left, right), method="bounded",
                options={"xatol": 1e-12},
            )
            score, coefficients = fit_at(float(refined.x))
            if best is None or score < best[0]:
                best = (score, float(refined.x), coefficients)
        if best is None:
            i = int(np.nanargmin(scores))
            best = (float(scores[i]), float(frequencies[i]), fit_at(float(frequencies[i]))[1])

        score, frequency, coefficients = best
        terms = ["1"] if include_bias else []
        terms.extend(
            f"({carrier_symbol.to_str()}) ** {power}" for power in range(1, degree + 1)
        )
        terms.append(
            f"({carrier_symbol.to_str()}) * sin(({frequency:.16g}) * ({oscillator_symbol.to_str()}))"
        )
        formula = " + ".join(
            f"({float(coefficient):.16g}) * ({term})"
            for coefficient, term in zip(coefficients, terms)
            if abs(coefficient) > 1e-14
        ) or "0"
        evaluation = self.evaluate(
            f=self.parse_formula(formula),
            y=target_symbol,
            show_diagnostics=show_diagnostics,
        )
        return evaluation | {
            "fit_configuration": {
                "carrier": carrier_symbol.to_str(),
                "oscillator": oscillator_symbol.to_str(),
                "degree": degree,
                "include_bias": include_bias,
                "frequency": frequency,
                "training_grid_mse": score,
            },
        }

    @classmethod
    def format_result_dict(cls, result: Dict[str, Any]) -> str:
        text = cls.format_evaluation_result(result, title="Fitted harmonic interaction")
        config = result["fit_configuration"]
        return text + (
            f"\nHarmonic interaction: carrier={config['carrier']}, "
            f"oscillator={config['oscillator']}, frequency={config['frequency']:.8g}."
        )
