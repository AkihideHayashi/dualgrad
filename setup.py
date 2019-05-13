from setuptools import setup, find_packages
# from Cython.Build import cythonize
# import numpy

setup(
    name="dualgrad",
    version="0.0.0",
    install_requires=["numpy", "scipy", "sympy"],
    packages=find_packages(),
    # ext_modules=cythonize("dualgrad/*.pyx", numpy.get_include())
)
