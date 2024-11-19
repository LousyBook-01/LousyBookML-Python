from setuptools import setup, find_packages

setup(
    name="lousybookml",
    version="0.1.0",
    packages=find_packages(where="lib"),
    package_dir={"": "lib"},
    install_requires=[
        "numpy>=1.19.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "matplotlib>=3.3.0",
            "pandas>=1.0.0",
        ]
    },
    author="LousyBook01",
    description="A Python machine learning library from scratch made by LousyBook01",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    keywords="machine learning, neural networks, linear regression, from scratch",
    python_requires=">=3.7",
)
