"""
simplestereo
============
Module initialization.

Documentation DOCSTRING follows numpy-style wherever possible.
See https://numpydoc.readthedocs.io/en/latest/format.html
"""

# Load main classes and allow direct use at package level.
from ._rigs import *

# Load submodules.
from . import calibration
from . import rectification
from . import passive
from . import active
from . import unwrapping
from . import points
from . import utils

### VERSION INFO
import pkg_resources # part of setuptools
__version__ = pkg_resources.require("SimpleStereo")[0].version


