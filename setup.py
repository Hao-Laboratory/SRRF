import setuptools
from srrf_cupy import __version__

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as rq:
    requirements = rq.read().splitlines()

setuptools.setup(
    name="SRRF-cupy",
    version=__version__,
    author="Lu Xiao",
    author_email="lx.combox@gmail.com",
    description="A package reconstructing super-resolution image using SRRF algorithm",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hak0/srrf_cupy",
    packages=setuptools.find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    scripts = ['srrf_cupy/srrf.py', 'srrf_cupy/ui_srrf.py'],
)
