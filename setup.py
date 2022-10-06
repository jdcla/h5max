import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="h5max",
    version="0.1.0",
    author="Jim Clauwaert",
    author_email="jim.clauwaert@ugent.be",
    packages=["h5max"],
    description="A package built upon h5py to facilitate data loading and saving",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jdcla/h5max",
    license='MIT',
    python_requires='>=3.9',
    install_requires=[
         "numpy>=1.21.0", "scipy>=1.7.3"
    ]
)
