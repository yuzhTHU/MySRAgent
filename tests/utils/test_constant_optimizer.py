import numpy as np
import nd2py as nd

from sr_agent.utils.constant_optimizer import ConstantOptimizerConfig, fit_constants


FAST_CONFIG = ConstantOptimizerConfig(
    n_restarts=2,
    max_nfev=300,
    use_global_search=False,
    snap_exponents=False,
)


def test_fit_constants_projects_affine_parameters_without_mutating_input():
    x = np.linspace(-2.0, 2.0, 101)
    expression = nd.parse("0.3*x + 0.7")
    before = expression.to_str(number_format=".16g")

    fitted = fit_constants(expression, {"x": x}, 2.5 * x - 1.2, config=FAST_CONFIG)

    assert fitted is not expression
    assert expression.to_str(number_format=".16g") == before
    assert np.mean((fitted.eval({"x": x}) - (2.5 * x - 1.2)) ** 2) < 1e-20


def test_fit_constants_handles_separable_nonlinear_parameters():
    x = np.linspace(-1.5, 1.5, 151)
    expression = nd.parse("0.2*sin(0.4*x + 0.1) + 0.3*x + 0.5")
    target = 1.7 * np.sin(2.1 * x - 0.35) - 0.8 * x + 0.25

    fitted = fit_constants(
        expression,
        {"x": x},
        target,
        config=ConstantOptimizerConfig(
            random_state=4,
            n_restarts=4,
            max_nfev=800,
            differential_evolution_maxiter=20,
            differential_evolution_popsize=5,
            snap_exponents=False,
        ),
    )

    assert np.mean((fitted.eval({"x": x}) - target) ** 2) < 1e-12


def test_fit_constants_uses_feature_domain_for_large_offsets():
    temperature = np.linspace(270.0, 390.0, 121)
    expression = nd.parse("0.4*(T - 0.2) + 0.1")
    target = 2.3 * (temperature - 347.0) - 4.2

    fitted = fit_constants(expression, {"T": temperature}, target, config=FAST_CONFIG)

    assert np.mean((fitted.eval({"T": temperature}) - target) ** 2) < 1e-18


def test_fit_constants_returns_copy_when_there_are_no_fitable_parameters():
    x = np.arange(5.0)
    expression = nd.Variable("x")

    fitted = fit_constants(expression, {"x": x}, x, config=FAST_CONFIG)

    assert fitted is not expression
    assert np.array_equal(fitted.eval({"x": x}), x)
