#!/usr/bin/env python

from distutils.core import setup
from setuptools import find_packages

setup(
    name="recsys-gnn",
    version="1.0",
    description="RecSys GNN recommender system",
    author="Yanina Kutovaya",
    author_email="kutovaiayp@yandex.ru",
    url="https://github.com/Yanina-Kutovaya/recsys-gnn.git",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
)