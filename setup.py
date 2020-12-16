#!/usr/bin/env python3
import codecs
import os

from setuptools import setup, find_packages


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    # intentionally *not* adding an encoding option to open, See:
    #   https://github.com/pypa/virtualenv/issues/201#issuecomment-3145690
    with codecs.open(os.path.join(here, rel_path), "r") as fp:
        return fp.read()


def get_version():
    return read("check_cuda/VERSION")


setup(
    # dependency_links=[
    #     'git+https://github.com/vtpl1/singleton_decorator.git@master#egg=singleton-decorator'
    # ],
    install_requires=[
        'singleton_decorator@git+https://github.com/vtpl1/singleton_decorator.git', 'py-cpuinfo', 'PyYAML',
        'dataclasses'
    ],
    name="check-cuda",
    version=get_version(),
    fullname="Get NVIDIA GPU devices",
    description="Get NVIDIA GPU devices",
    author="Monotosh Das",
    author_email="monotosh.das@videonetics.com",
    keywords="cuda cpu hardware",
    long_description=open('README.md').read(),
    url="https://github.com/vtpl1/check_cuda",
    license="MIT",
    include_package_data=True,
    packages=find_packages(exclude=["*.tests", "test", "session"]),
    # package_dir={'negar': 'negar'},
    package_data={'': ['*.yaml', 'VERSION']},
    entry_points={
        'console_scripts': [
            'lopymo = lopymo.main:main',
        ],
    })
