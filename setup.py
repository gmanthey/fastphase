import os
import numpy
from setuptools import Extension, setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

#### Deal with Cython
try:
    import Cython
    cython = True
except ImportError:
    cython = False

ext = '.pyx' if cython else '.c'

extensions=[
    Extension('fastphase.fastphase',
              sources = ["fastphase/fastphase"+ext],
              include_dirs=[numpy.get_include()]
              ),
    Extension('fastphase.calc_func',
              sources = ["fastphase/calc_func"+ext],
              include_dirs=[numpy.get_include()]
              )
    ]

if cython:
    from Cython.Build import cythonize
    extensions = cythonize(extensions, include_path=[numpy.get_include()])


setup(
    packages = ['fastphase'],
    package_data={'fastphase':["*.pyx"]},
    ext_modules = extensions
    )
