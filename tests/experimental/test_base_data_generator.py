import numpy as np
import pytest

nd = pytest.importorskip("nd2py")

from src.experimental.nn_tools.datasets.generate_data import BaseDataGenerator


@pytest.mark.parametrize("name", ["gaussian", "uniform", "gmm"])
def test_data_generators_return_1d_arrays(name):
    generator = BaseDataGenerator.create(name, sample_num=16, random_seed=123)

    data_dict, target, success = generator(nd.parse("x0+x1"))

    assert set(data_dict) == {"x0", "x1"}
    assert all(value.shape == (16,) for value in data_dict.values())
    assert target.shape == (16,)
    assert success
    assert np.isfinite(target).all()


def test_data_generator_broadcasts_constant_expression():
    generator = BaseDataGenerator.create("uniform", sample_num=8, random_seed=123)

    data_dict, target, success = generator(nd.Number(3.0))

    assert data_dict == {}
    assert target.shape == (8,)
    assert np.all(target == 3.0)
    assert success


def test_data_generator_accepts_zero_random_seed():
    generator_a = BaseDataGenerator.create("uniform", sample_num=4, random_seed=0)
    generator_b = BaseDataGenerator.create("uniform", sample_num=4, random_seed=0)

    data_a, target_a, success_a = generator_a(nd.parse("x0"))
    data_b, target_b, success_b = generator_b(nd.parse("x0"))

    assert generator_a.random_seed == 0
    assert generator_b.random_seed == 0
    assert success_a and success_b
    np.testing.assert_array_equal(data_a["x0"], data_b["x0"])
    np.testing.assert_array_equal(target_a, target_b)
