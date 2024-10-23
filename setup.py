"""Module setup."""

import runpy
from setuptools import setup, find_packages

PACKAGE_NAME = "tail_test"
# version_meta = runpy.run_path("./version.py")
# VERSION = version_meta["__version__"]
VERSION = "0.0.5"

with open("PYPI_README.md", "r") as fh:
    long_description = fh.read()

def parse_requirements(filename):
    """Load requirements from a pip requirements file."""
    lineiter = (line.strip() for line in open(filename))
    return [line for line in lineiter if line and not line.startswith("#")]

if __name__ == "__main__":
    setup(
        name=PACKAGE_NAME,
        version=VERSION,
        packages=find_packages(),
        install_requires=parse_requirements("requirements.txt"),
        python_requires=">=3.8.3",
        scripts=["scripts/tail-cli.build","scripts/tail-cli.eval"],
        description="A Toolkit for automatic LLM Evaluation.",
        long_description=long_description,
        long_description_content_type="text/markdown",
    )