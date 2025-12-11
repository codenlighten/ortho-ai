from setuptools import setup, find_packages

setup(
    name="okadfa",
    version="0.1.0",
    description="Orthogonalized Kernel Attention with Direct Feedback Alignment for Efficient LLM Training",
    author="NeuroLab AI Syndicate",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "numpy>=1.24.0",
        "einops>=0.7.0",
        "transformers>=4.35.0",
        "accelerate>=0.25.0",
        "datasets>=2.15.0",
        "scipy>=1.11.0",
        "wandb>=0.16.0",
        "pyyaml>=6.0",
        "omegaconf>=2.3.0",
        "tqdm>=4.66.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.1.0",
        ],
    },
)
