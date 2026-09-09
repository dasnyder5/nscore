from setuptools import find_packages, setup

setup(
    name="nscore",
    version="0.1.2",
    description="Sequential statistical hypothesis testing for generalized performance measures.",
    author="David Snyder, Haruki Nishimura",
    author_email="dsnyder5@engineering.upenn.edu, haruki.nishimura@tri.global",
    packages=find_packages(),
    install_requires=[
        "matplotlib",
        "numpy>=1.20",
        "scipy",
        "cvxpy",
        "statistical-comparison-core>=0.2.1,<0.3",
        "statistical-comparison-helpers",
    ],
)
