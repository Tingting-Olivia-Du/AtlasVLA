"""
Setup script for AtlasVLA project
"""

from setuptools import setup, find_packages
import os

# Read requirements
def read_requirements():
    req_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
    if os.path.exists(req_path):
        with open(req_path, "r") as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return []

setup(
    name="atlas-vla",
    version="0.1.0",
    description="Vision-Language-Action model based on VGGT for robot manipulation",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/AtlasVLA",
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    install_requires=read_requirements(),
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
