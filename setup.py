from setuptools import setup
from setuptools import find_packages

setup(
    name="svd",
    version="0.0.1",
    author="mirucaaura",
    description="python implementation of svd",
    packages=find_packages("src"),
    package_dir={"": "src"},
    python_requires='>=3.7',
)