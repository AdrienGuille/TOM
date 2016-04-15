# coding: utf-8
from setuptools import setup, find_packages
import tom_lib

__author__ = "Adrien Guille, Pavel Soriano"
__email__ = "adrien.guille@univ-lyon2.fr"

setup(
    name='tom_lib',
    version=tom_lib.__version__,
    packages=find_packages(),
    author="Adrien Guille, Pavel Soriano",
    author_email="adrien.guille@univ-lyon2.fr",
    description="A library for topic modeling and browsing",
    long_description=open('README.rst').read(),
    url='http://mediamining.univ-lyon2.fr/people/guille/tom.php',
    classifiers=[
        'Programming Language :: Python',
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        "Operating System :: OS Independent",
        'Programming Language :: Python :: 2.7',
        'Topic :: Scientific/Engineering'
    ])