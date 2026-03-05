#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GRA-Swarm Setup Script
Скрипт Установки Пакета
"""

from setuptools import setup, find_packages
from pathlib import Path

# Чтение README / Read README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Чтение зависимостей / Read dependencies
requirements = []
req_file = this_directory / "requirements.txt"
if req_file.exists():
    with open(req_file, "r", encoding="utf-8") as f:
        requirements = [
            line.strip() 
            for line in f 
            if line.strip() and not line.startswith("#")
        ]

setup(
    name="gra-swarm",
    version="0.1.0",
    author="YOUR_NAME",
    author_email="your.email@example.com",
    description="Swarm Superintelligence via Cognitive Foam Minimization / "
                "Роевой Сверхинтеллект через Минимизацию Когнитивной Пены",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/YOUR_USERNAME/GRA-Swarm",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "black>=21.0",
            "flake8>=3.9.0",
            "mypy>=0.910",
            "pre-commit>=2.13.0",
            "pytest>=6.2.0",
            "pytest-cov>=2.12.0",
        ],
        "docs": [
            "mkdocs>=1.2.0",
            "mkdocs-material>=7.0.0",
        ],
        "llm": [
            "openai>=0.27.0",
            "transformers>=4.20.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "gra-swarm=gra_swarm.cli:main",
        ],
    },
)