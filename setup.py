import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup

extensions=[
    Extension('fastphase.fastphase',
              sources = ["fastphase/fastphase.pyx"],
              include_dirs=[numpy.get_include()]
              ),
    Extension('fastphase.calc_func',
              sources = ["fastphase/calc_func.pyx"],
              include_dirs=[numpy.get_include()]
              )
    ]

extensions = cythonize(extensions, include_path=[numpy.get_include()])


setup(
    name='fastphase',
    ext_modules = extensions
    )
