from setuptools import setup
import os

# Read the contents of the README file
with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="daytona-openai-demo",
    version="0.1.0",
    description="Enhanced OpenAI client with Daytona sandbox execution capabilities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Bruno Grbavac",
    author_email="info@daytona.io",
    url="https://github.com/brunogrbavac/daytona-openai",
    py_modules=["daytona_openai_demo"],
    install_requires=[
        "openai>=1.0.0",
        "daytona-sdk",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.10",
)