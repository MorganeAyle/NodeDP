# cython: language_level=3
from distutils.core import setup
from Cython.Build import cythonize
import numpy

import os
os.environ["CC"] = "g++"
os.environ["CXX"] = "g++"

setup(ext_modules = cythonize(["src/sampling/cython_sampler.pyx","src/sampling/cython_utils.pyx"]), include_dirs = [numpy.get_include()])
# to compile: python src/sampling/setup.py build_ext --inplace
