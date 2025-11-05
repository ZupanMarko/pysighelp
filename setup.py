from setuptools import setup, find_packages

setup(
    name="pysighelp",
    version="0.1.2",
    packages=find_packages(include=["pysighelp", "pysighelp.*"]),
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
    ],
    author="Marko Zupan",
    author_email="marko.zupan@fs.uni-lj.si",
    description="Signal processing helper functions in Python",
    url="https://github.com/ZupanMarko/pysighelp",
    python_requires=">=3.8",
)
