#!/usr/bin/env python

from setuptools import setup
from pip._internal import main as pipmain


def read_text_file(filename: str):
    with open(filename, "r") as text_file:
        return text_file.read()

long_description = read_text_file("README.md")
requirements = read_text_file("requirements.txt").split("\n")

pipmain(['install', 'https://github.com/evil-mad/axidraw/releases/download/v3.7.0/AxiDraw_API_320.zip'])

setup(
    name='axifresco',
    version='0.1',
    description='A python toolkit for drawing Fresco-format files with an EvilMadScientist Axidraw',
    long_description=long_description,
    author='Niels Pichon',
    author_email='niels.pichon@outlook.com',
    url='https://github.com/NielsPichon/AxiFresco.git',
    install_requires=requirements,
    python_requires=">=3.8",
    packages=['axifresco'],
    entry_points={"console_scripts": ["axifresco = axifresco.__main__:main"]},
)
