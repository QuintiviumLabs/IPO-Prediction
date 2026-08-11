import numpy as np
import pytest

from ipo_model.config import Config
from ipo_model.data.features import build_features, load_raw
from ipo_model.data.synthetic import generate


@pytest.fixture(scope="session")
def small_data(tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    generate(d, n_ipos=300, start="2015-01-05", end="2020-12-31", seed=11)
    return d


@pytest.fixture(scope="session")
def cfg():
    return Config()


@pytest.fixture(scope="session")
def fs(small_data, cfg):
    return build_features(load_raw(small_data), cfg.data)


@pytest.fixture(scope="session")
def raw(small_data):
    return load_raw(small_data)


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(0)
