import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="flatds",
    version="0.0.1",
    author="Tobias Kölling",
    author_email="tobias.koelling@physik.uni-muenchen.de",
    description="FlatDS - structured arrays, optimized for sequential writing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/d70-t/flatds",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3',
    install_requires=[
        "numpy",
        "xarray",
        "msgpack",
    ],
)
