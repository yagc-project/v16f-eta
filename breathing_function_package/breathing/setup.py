"""
Setup for Breathing Function Package
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="breathing-function",
    version="0.1.0",
    author="YAGC Project",
    author_email="contact@taiwacosmos.com",
    description="A universal activation function for adaptive systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/taiwacosmos/breathing",
    project_urls={
        "Website": "https://taiwacosmos.com",
        "Papers": "https://zenodo.org/communities/yagc",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Games/Entertainment",
        "Topic :: Multimedia :: Sound/Audio",
    ],
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
    ],
    extras_require={
        "dev": [
            "matplotlib>=3.5.0",
            "pytest>=7.0.0",
        ],
    },
    keywords=[
        "activation-function",
        "adaptive-systems",
        "breathing",
        "cosmology",
        "game-development",
        "ui-ux",
        "neural-networks",
    ],
)
