import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()

with open('misc/requirements_torch.txt', 'r', encoding='utf-16') as f:
    required = f.read().splitlines()
required += ["torchstain@git+https://github.com/andreped/torchstain@main"]

setup(
    name='fastStainNorm',
    version='1.0.0',
    description='Package for fast color normalization of H&E-stained images',
    long_description=README,
    long_description_content_type='text/markdown',
    url='https://github.com/andreped/fast-stain-normalization',
    author='andreped',
    author_email='andrped94@gmail.com',
    license='MIT',
    zip_safe=False,
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'faststainnorm = examples.__main__:main'
        ]
    },
    install_requires=required,
    python_requires='>=3.6',
)