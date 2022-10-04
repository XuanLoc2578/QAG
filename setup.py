"""A setuptools based setup module.

See:
"""
from setuptools import setup

required = []

setup(
    name="qag_pegasus",
    # Versions should comply with PEP440. For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version="1.0.0",
    description="Question Answer Generation",
    author="Nguyen Xuan Loc",
    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=["Programming Language :: Python :: 3.7", ],
    package_data={"qag_pegasus": []},
    include_package_data=True,
    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=required,
    # List additional groups of dependencies here (e.g. development
    # dependencies). You can install these using the following syntax,
    # for example:
    # $ pip install -e .[dev,test]
    extras_require={"dev": [], "test": [], },
    py_modules=["qag_pegasus"],
)
