from dMG._version import __version__
from dMG.core import calc, data, post, utils
from dMG.core.data import loaders, samplers
from dMG.models import criterion, delta_models, neural_networks, phy_models
from dMG.models.model_handler import ModelHandler

# In case setuptools scm says version is 0.0.0
assert not __version__.startswith('0.0.0')

__all__ = [
    '__version__',
    'calc',
    'data',
    'post',
    'utils',
    'loaders',
    'samplers',
    'criterion',
    'delta_models',
    'neural_networks',
    'phy_models',
    'ModelHandler',
]
