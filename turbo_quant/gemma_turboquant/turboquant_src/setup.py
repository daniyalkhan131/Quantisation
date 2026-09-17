from setuptools import setup, find_packages

setup(
    name="turboquant",
    version="0.1.0",
    description="First open-source implementation of TurboQuant (arXiv 2504.19874) for LLM KV cache compression",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Vivek Varikuti",
    url="https://github.com/vivekvarikuti/turboquant",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0",
        "scipy>=1.10",
        "transformers>=4.43",
    ],
    extras_require={
        "dev": ["pytest"],
        "bnb": ["bitsandbytes", "accelerate"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
