from setuptools import setup, find_packages

setup(
    name="inca_agent",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "openai>=1.3.0",
        "anthropic>=0.3.0",
        "google-generativeai>=0.3.0",
        "mistralai>=0.0.7",
        "transformers>=4.35.0",
        "torch>=2.1.0",
        "pandas>=2.1.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "pytest>=7.4.0",
        "jupyter>=1.0.0",
        "scikit-learn>=1.3.0",
        "python-dotenv>=1.0.0"
    ],
    python_requires=">=3.8",
    author="Your Name",
    author_email="your.email@example.com",
    description="InCA (In-context Continual Learning) System",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
