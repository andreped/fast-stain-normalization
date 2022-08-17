import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()

with open('misc/requirements_torch.txt', 'r', encoding='utf-16') as f:
    required = f.read().splitlines()

setup(
    name='faststainnorm',
    version='1.0.0',
    description='Package for fast color normalization of H&E-stained images',
    long_description=README,
    long_description_content_type='text/markdown',
    url='https://github.com/andreped/fast-stain-normalization',
    author='André Pedersen',
    author_email='andrped94@gmail.com',
    license='MIT',
    zip_safe=False,
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'faststainnorm = source.__main__:main'
        ]
    },
    classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
    install_requires=required,
    python_requires='>=3.6',
)