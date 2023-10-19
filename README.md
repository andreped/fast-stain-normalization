# fast-stain-normalization

[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/andreped/fast-stain-normalization/workflows/CI/badge.svg)](https://github.com/andreped/fast-stain-normalization/actions)

This repository contains a tool for stain normalizing images, relevant for computational pathology.

We also include a minor runtime benchmark of some **open-source stain normalization** methods.

The source code is available for both. See [here](https://github.com/andreped/fast-stain-normalization#usage) for how to use the tool on your own data, and see [here](https://github.com/andreped/fast-stain-normalization#running-experiments) for how to redo the benchmark experiments.

![Screenshot](figures/example_subplot.PNG)

## [Install](https://github.com/andreped/fast-stain-normalization/tree/main#install)

```
pip install git+https://github.com/andreped/fast-stain-normalization.git
```

Note that to use the GPU, you will need CUDA installed and this requires a different install
method for the dependencies torch and torchvision. For instance, if you have CUDA 11.2 installed, run:

```
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```

Note that this will overwrite the currently installed `torch` and `torchvision` versions.

## [Usage](https://github.com/andreped/fast-stain-normalization/tree/main#usage)

```
faststainnorm --ref full-path-to-reference-image --img path-to-images-to-convert --out path-to-store-output
```

| command | description |
| ------------- | ------------- |
| `--ref` | the full path to the reference image you wish to use, including filename and format. |
| `--img` | should either be the path to the directly containing the images you wish to normalize, but it could also be the full path to a single image. |
| `--out` | the path to where you wish to store the result. It will be stored in the same structure as provided in `--img`, and default is `./`. |
| `--cpu` | to force computations to use the CPU. GPU disabled by default (=1). |
| `--mp` | to enable multiprocessing for performing batch mode with parallel processing. Disabled by default (=0). |
| `--wk` | set number of workers (relevant for multiprocessing). Default is 1. |

## [Experiment](https://github.com/andreped/fast-stain-normalization/tree/main#experiment)

For the benchmarking we used the libraries: [ITKColorNormalization](https://github.com/InsightSoftwareConsortium/ITKColorNormalization), [StainTools](https://github.com/Peter554/StainTools), and [torchstain](https://github.com/EIDOSlab/torchstain). TorchStain (TS) included implementation of the Macenko algorithm, ITK had the same but also Vahadane. Lastly, StainTools included both algorithms but also Reinhard. Runtime experiments were conducted over 50 iterations using default parameters, and the mean and standard deviation were reported. The hardware used was an eight-core Intel i7-9800X CPU, with 32 GB RAM, using the Ubuntu Linux 18.04 operating system.

Apriori, we know that the Vahadane method is the best performing method among the three. Macenko is faster than Vahadane, but less robust. Reinhard is the oldest methods of the three and less suited for H&E-stained images.

## [Result](https://github.com/andreped/fast-stain-normalization/tree/main#result)

| Method  | Reinhard | Macenko | Vahadane | Vahadane (ITK) | Macenko (TS) |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| Runtime (s)  | 0.0025 &pm; 0.0001 | 0.6169 &pm; 0.0165 | 1.4575 &pm; 0.0081 | 0.0449 &pm; 0.02706 | 0.0066 &pm; 0.0015 |

Preliminary results showed that the Reinhard color augmentation algorithm was the fastest, but the second fastest method was the Macenko implementation in TS. The third fastest was ITK's implementation of Vahadane. StainTools' implementations of Vahadane and Macenko fell short compared to its counterparts.

## [Discussion](https://github.com/andreped/fast-stain-normalization/tree/main#discussion)

After running ITK's implementation of Vahadane on other images, we found that the method was less robust than TS's implementation of Macenko. ITK seemed to crash often and produce errors on most images that contained either some noise or in scenarios where there were poor contrast between the colours, which might happen when looking at the tissue from a lower resolution level or from patches with mostly glass. Hence, the best trade-off among the three might be TS' implementation of Macenko as it is robust, fast, and provides suitable normalization performance [(see here for reference)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7460968).

## [Conclusion](https://github.com/andreped/fast-stain-normalization/tree/main#conclusion)

Even though StainTools offers more functionality, both in terms of stain normalization and augmentation, it is not nearly as optimized as the two other alternatives, for respective methods. We found that TS' implementation of Macenko to be best suited for our application, and we have implemented a tool for applying this on your own data.

## [Applications](https://github.com/andreped/fast-stain-normalization#applications)
* Alharbi et al., A Deep Learning–Based Approach to Estimate Paneth Cell Granule Area in Celiac Disease (2023), https://doi.org/10.5858/arpa.2023-0074-OA

## [Running Experiments](https://github.com/andreped/fast-stain-normalization/tree/main#running-experiments)

1. Clone repository:
```
git clone https://github.com/andreped/fast-stain-normalization.git
cd fast-stain-normalization/
```

2. Create virtual environment and install dependencies:
```
virtualenv -ppython3 venv --clear
source venv/bin/activate
pip install -r misc/requirements.txt
cd examples/
```

3. Run script using the CLI:
```
python runtime.py
```

The script support various arguments, i.e. for choosing which method to use or how many iterations to run. Append **-h** to the command to see which arguments are supported and how to use them.

## [Troubleshooting](https://github.com/andreped/fast-stain-normalization/tree/main#troubleshooting)

Virtualenv can be installed using pip:
```
pip install virtualenv
```

To activate virtual environments on Windows (the description above was for Unix systems), you can run the command:
```
./venv/Scripts/activate
```

#### [Benchmark-related only](https://github.com/andreped/fast-stain-normalization/tree/main#benchmark-related-only)
Note that StainTools depends on [SPAMS](https://github.com/samuelstjean/spams-python), which is currently not supported on Windows. Hence, it would not be possible to run the experiments using the Windows operating system. However, Ubuntu Linux and macOS should work.

If the patch size chosen is too small, the stain normalization methods might fail, especially the ITK-implementation. The same implementation might also fail if a colourless patch is provided. The reference image and input images should therefore be of representative size with meaningful content, for the method to produce a meaningful output.


## [Acknowledgements](https://github.com/andreped/fast-stain-normalization/tree/main#acknowledgements)

This could not have been possible without the great effort of fellow open-source GitHub users that provide brilliant solutions for me to test and explore!

The code in this repository is based on the three GitHub repositories: [ITKColorNormalization](https://github.com/InsightSoftwareConsortium/ITKColorNormalization), [StainTools](https://github.com/Peter554/StainTools), and [torchstain](https://github.com/EIDOSlab/torchstain). Where the latter, torchstain, was used as foundation to develop the tool.
