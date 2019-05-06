import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="flatds-tobi", # Replace with your own username
    version="0.0.1",
    author="Tobias Kölling",
    author_email="tobias.koelling@physik.uni-muenchen.de",
    description="FlatDS - structured arrays, optimized for sequential writing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/d70-t/flatds",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=2.7, >=3.0',
    install_requires=[
        "numpy",
        "xarray",
        "msgpack",
    ],
)