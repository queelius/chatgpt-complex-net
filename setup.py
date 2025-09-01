#!/usr/bin/env python3
"""
Setup script for llm-semantic-net package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="llm-semantic-net",
    version="0.1.0",
    author="Alex Towell",
    description="A toolkit for generating semantic networks from various data sources",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/queelius/llm-semantic-net",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "semnet=cli:main",
            "semnet-rec=rec_conv:main",
        ],
    },
    include_package_data=True,
)