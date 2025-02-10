from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="metasim",
    version="0.0.1",
    author="Jieming Li",
    author_email="jmli@umich.edu",
    description="A foundation model for single-molecule time traces",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/simol-lab/META-SiM",
    project_urls={
        "Bug Tracker": "https://github.com/simol-lab/META-SiM/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Physics",
        "Intended Audience :: Science/Research",
    ],
    install_requires=[
        "keras>=2.15.0",
        "tensorflow>=2.15.0",
        "umap-learn>=0.5.3",
        "numpy",
        "scipy",
        "sklearn",
    ],
)
