# type: ignore

from setuptools import setup, find_packages

setup(
    name="knnopt",
    version="0.1.0",
    description="A modular Python toolkit for benchmarking and visualizing optimal K values in KNN classification.",
    author="Guy Kaptue",
    author_email="guykaptue24@gmail.com",
    license="MIT",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.8.0",
        "tqdm>=4.66.1"
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.3.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
            "jupyter>=1.0.0"
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)

