import os
from setuptools import setup
from setuptools.extension import Extension

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

#### Deal with Cython
try:
    import Cython
    cython = True
except ImportError:
    cython = False

ext = '.pyx' if cython else '.c'

#### Deal with numpy
#### see https://stackoverflow.com/questions/54117786

# factory function
def my_build_ext(pars):
    # import delayed:
    from setuptools.command.build_ext import build_ext as _build_ext#
    # include_dirs adjusted: 
    class build_ext(_build_ext):
        def finalize_options(self):
            _build_ext.finalize_options(self)
            # Prevent numpy from thinking it is still in its setup process:
            __builtins__.__NUMPY_SETUP__ = False
            import numpy
            self.include_dirs.append(numpy.get_include())
    #object returned:
    return build_ext(pars)

extensions=[
    Extension('fastphase.fastphase',
              sources = ["fastphase/fastphase"+ext])
    ]

if cython:
    from Cython.Build import cythonize
    extensions = cythonize(extensions)


setup(
    name = 'fastphase',
    version = '1.2',
    description = 'Python implementation of the fastPHASE model',
    long_description = read('README.md'),
    license = "LGPL v3",
    author = "Bertrand Servin",
    author_email = "bertrand.servin@inrae.fr",
    url = "https://forgemia.inra.fr/bertrand.servin/fastphase",
    packages = ['fastphase'],
    cmdclass={'build_ext' : my_build_ext},
    setup_requires = [ 'numpy' ],
    install_requires = [
        'numpy',
        'scipy'
        ],
    ext_modules = extensions,
    classifiers = [
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        "Operating System :: OS Independent",
        "Programming Language :: Cython",
        "Topic :: Scientific/Engineering"
        ]
    )
