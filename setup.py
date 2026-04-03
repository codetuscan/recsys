"""Setup script for recsys package."""

from setuptools import setup

setup(
    name="recsys",
    version="0.1.0",
    description="GPU-accelerated recommender system for breaking filter bubbles",
    author="Your Name",
    packages=[
        "recsys",
        "recsys.config",
        "recsys.data",
        "recsys.evaluation",
        "recsys.experiments",
        "recsys.models",
        "recsys.utils",
    ],
    python_requires=">=3.10",
    install_requires=[
        "numpy>=2.0",
        "pandas>=3.0",
        "torch>=2.0",
        "PyYAML>=6.0",
        "scikit-learn>=1.0",
        "scipy>=1.17",
        "matplotlib>=3.0",
        "seaborn>=0.13",
        "tqdm>=4.0",
    ],
)
