from setuptools import setup, find_packages

setup(
    name="oncoresolve",
    version="1.0.0",
    author="Shubham Jha",
    author_email="shubhamkjha369@gmail.com",
    description="A High-Hygiene Explainable AI and Patient-Centric Uniqueness Framework for Breast Cancer Subtyping",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/shubhamkjha369/OncoResolve-Breast-Cancer-Transcriptomics",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "joblib>=1.3.0",
        "lifelines>=0.27.0",
    ],
)
