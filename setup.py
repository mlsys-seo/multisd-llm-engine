from setuptools import setup, find_packages
# from torch.utils.cpp_extension import BuildExtension, CUDAExtension

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# # Read requirements from requirements.txt
# with open("requirements.txt", "r", encoding="utf-8") as f:
#     install_requires = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="sdprofiler",
    version="0.1.0",
    author="Sungkyun Kim",
    author_email="sung.balance@gmail.com",
    description="A fast and efficient library for working with large language models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sung.balance/sdprofiler",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
)
