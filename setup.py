from setuptools import setup, find_packages

setup(
    name="metameta",
    version="0.1.0",
    author="Walter Obiero",
    author_email="obierochieng@yahoo.com",
    description="A Python package for conducting meta-analysis",
    long_description=open("ReadMe.md").read(),
    long_description_content_type="text/markdown",
    url="https://github/wobiero/meta_analyzer",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "matplotlib"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved : MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.6",
)