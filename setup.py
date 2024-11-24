"""
LousyBookML Setup Configuration
"""

from setuptools import setup, find_packages

setup(
    name="lousybook01.LousyBookML",
    version="0.5.0-beta",
    author="LousyBook01",
    author_email="lousybook94@gmail.com",
    description="A comprehensive machine learning library for educational purposes",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/LousyBook-01/LousyBookML-Python",
    packages=find_packages(where="lib"),
    package_dir={"": "lib"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Anyone",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24.0",
        "gymnasium>=0.29.0",
        "pygame>=2.5.0",
        "matplotlib>=3.7.1",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.1",
            "pytest-cov>=4.1.0",
        ],
    },
)
