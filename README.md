# fast-stain-normalization

This repository contains a Python script for estimating runtime of some **open-source stain normalization** methods. For the benchmarking we used the libraries: [ITKColorNormalization](https://github.com/InsightSoftwareConsortium/ITKColorNormalization) and [StainTools](https://github.com/Peter554/StainTools).

## Experiment

ITK included an implementation of the Vahadane algorithm, whereas StainTools included implementations of the Vahadane, Macenko, and Reinhard algorithm. Runtime experiments were conducted over 50 iterations using default parameters, and the mean and standard deviation were reported. The hardware used was an eight-core, Intel i7-9800X CPU, with 64 GB RAM, using the Ubuntu Linux 18.04 operating system.

## Result

| Method  | Reinhard | Macenko | Vahadane | Vahadane (ITK) |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Runtime (s)  | 0.00251 +- 0.00007 | 0.61686 +- 0.01654 | 1.45747 +- 0.00810 | 0.04486 +- 0.027056 |

Preliminary results show that the Reinhard color augmentation algorithm was the fastest. However, it is known that this method performs much poorer compared to the other alternatives. Surprisingly, the second fastest method was ITK's implementation of Vahadane, which was the slowest method in StainTools.

## Conclusion

Even though StainTools provides a wider range of functionality than ITK, both in terms of stain normalization and augmentation, it is not nearly as optimized as the ITK alternative (using the Vahadane algorithm). Therefore, for stain normalization purposes using the Vahadane algorithm, one should use the ITK alternative.

