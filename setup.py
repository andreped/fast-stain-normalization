import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()

with open('misc/requirements_torch.txt') as f:
    required = f.read().splitlines()

setup(
    name='fastStainNorm',
    version='1.0.0',
    description='Fast color normalization tool for H&E-stained images',
    long_description=README,
    long_description_content_type='text/markdown',
    url='https://github.com/andreped/fast-stain-normalization',
    author='andreped',
    author_email='andrped94@gmail.com',
    license='MIT',
    packages=find_packages(exclude=('tests')),
    zip_safe=False,
    install_requires=[
        required,
        "git+https://github.com/andreped/torchstain.git",
    ],
    python_requires='>=3.6'
)