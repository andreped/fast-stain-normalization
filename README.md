# fast-stain-normalization

This repository contains a Python script for estimating runtime of some **open-source stain normalization** methods.

## Experiment

For the benchmarking we used the libraries: [ITKColorNormalization](https://github.com/InsightSoftwareConsortium/ITKColorNormalization) and [StainTools](https://github.com/Peter554/StainTools). ITK included an implementation of the Vahadane algorithm, whereas StainTools included implementations of the Vahadane, Macenko, and Reinhard algorithms. Runtime experiments were conducted over 50 iterations using default parameters, and the mean and standard deviation were reported. The hardware used was an eight-core, Intel i7-9800X CPU, with 64 GB RAM, using the Ubuntu Linux 18.04 operating system.

## Result

| Method  | Reinhard | Macenko | Vahadane | Vahadane (ITK) |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Runtime (s)  | 0.00251 +- 0.00007 | 0.61686 +- 0.01654 | 1.45747 +- 0.00810 | 0.04486 +- 0.027056 |

Preliminary results showed that the Reinhard color augmentation algorithm was the fastest. However, it is known that this method performs much poorer on histopathological images compared to the other alternatives. Surprisingly, the second fastest method was ITK's implementation of Vahadane, which was the slowest method in StainTools.

## Conclusion

Even though StainTools offers more functionality than ITKColorNormalization for histopathological applications, in terms of stain normalization and augmentation, it is not nearly as optimized as the ITK alternative (at least not their Vahadane implementation). Therefore, for stain normalization purposes, one should use the ITK alternative, as the Vahadane method is known to be the best performing method among the three.

## Usage

1. Clone repository:
```
git clone https://github.com/andreped/fast-stain-normalization.git
cd fast-stain-normalization/
```

2. Create virtual environment and install dependencies:
```
virtualenv -ppython3 venv --clear
pip install -r requirements.txt
cd examples/
```

3. Run script using the CLI:
```
python runtime.py
```

The script support various arguments, i.e. for choosing which method to use or how many iterations to run. Append **-h** to the command to see which arguments are supported and how to use them.

