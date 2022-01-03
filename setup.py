import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="b4",
    version="0.0.1",
    author="Aaron Walsman",
    author_email="aaronwalsman@gmail.com",
    description='A worked solution to the Bodega Brawl card game.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aaronwalsman/b4",
    install_requires = [
        'numpy',
        'scipy',
        'tqdm',
    ],
    packages=setuptools.find_packages(),
    entry_points = {
    },
    classifiers = [
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ]
)
