from setuptools import setup
import os
import sys

_here = os.path.abspath(os.path.dirname(__file__))

setup(
    name="imblanced",
    version="0.0",
    description=("imbalanced data"),
    long_description='',
    license="MPL-2.0",
    packages=["imbalanced"],
    install_requires=[
        "tqdm==4.26.0",
        "numpy>=1.14.3",
        "torchvision>=0.2.1",
        "gpytorch>=0.1.0rc4",
        "tabulate>=0.8.2",
        "scipy>=1.1.0",
        "setuptools>=39.1.0",
        "matplotlib>=2.2.2",
        "torch>=1.0.0",
        "Pillow>=5.4.1",
        "scikit_learn>=0.20.2",
    ],
    include_package_data=True,
    classifiers=[
        "Development Status :: 0",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.6",
    ],
)
