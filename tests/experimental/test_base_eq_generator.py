import pytest

nd = pytest.importorskip("nd2py")

from src.experimental.nn_tools.datasets.generate_eq import BaseEqGenerator, GplearnEqGenerator


def test_gplearn_eq_generator_generates_scalar_expression_with_n_var():
    generator = GplearnEqGenerator(
        n_variables=3,
        random_seed=123,
        const_range=(-1.0, 1.0),
        depth_range=(2, 4),
        n_var_range=(3, 4),
    )

    eq = generator.generate_eq()
    variable_names = {symbol.name for symbol in eq.iter_preorder() if isinstance(symbol, nd.Variable)}

    assert isinstance(eq, nd.Symbol)
    assert variable_names <= {"x0", "x1", "x2"}


def test_gplearn_eq_generator_accepts_configured_variables():
    generator = GplearnEqGenerator(
        variables=["a", "b"],
        random_seed=123,
        const_range=(-1.0, 1.0),
        depth_range=(2, 4),
    )

    eq = generator()
    variable_names = {symbol.name for symbol in eq.iter_preorder() if isinstance(symbol, nd.Variable)}

    assert variable_names <= {"a", "b"}


def test_gplearn_eq_generator_is_registered():
    generator = BaseEqGenerator.create(
        "gplearn",
        n_variables=1,
        random_seed=123,
        const_range=(-1.0, 1.0),
        depth_range=(2, 4),
        n_var_range=(1, 2),
    )

    assert isinstance(generator.generate_eq(), nd.Symbol)


def test_eq_generator_accepts_zero_random_seed():
    generator_a = BaseEqGenerator.create(
        "gplearn",
        n_variables=1,
        random_seed=0,
        const_range=(-1.0, 1.0),
        depth_range=(2, 4),
        n_var_range=(1, 2),
    )
    generator_b = BaseEqGenerator.create(
        "gplearn",
        n_variables=1,
        random_seed=0,
        const_range=(-1.0, 1.0),
        depth_range=(2, 4),
        n_var_range=(1, 2),
    )

    assert generator_a.random_seed == 0
    assert generator_b.random_seed == 0
    assert str(generator_a.generate_eq()) == str(generator_b.generate_eq())


@pytest.mark.parametrize("name", ["metaai", "snip", "snip2", "snip3"])
def test_eq_generators_are_registered(name):
    generator = BaseEqGenerator.create(
        name,
        n_variables=2,
        random_seed=123,
        n_operators=4,
        max_unary=2,
        max_var=2,
    )

    eq = generator.generate_eq()
    variable_names = {symbol.name for symbol in eq.iter_preorder() if isinstance(symbol, nd.Variable)}

    assert isinstance(eq, nd.Symbol)
    assert variable_names <= {"x0", "x1"}
