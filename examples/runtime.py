from urllib.request import urlretrieve
import os
import itk
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import staintools
import argparse
from tqdm import tqdm
import sys
import torch
from torchvision import transforms
import torchstain


def normalize(input_image, reference_image, method, both, normalizer_m):
    if method == "staintools":
        if normalizer_m in ["macenko", "vahadane"]:
            normalizer = staintools.StainNormalizer(method=normalizer_m)
        elif normalizer_m == "reinhard":
            normalizer = staintools.ReinhardColorNormalizer()
        else:
            raise ValueError

        input_image = np.asarray(input_image)
        reference_image = np.asarray(reference_image)

        if both:
            time_ = time.time()
            normalizer.fit(reference_image)
            transformed = normalizer.transform(input_image)  # only report runtime for transform (not considering fit to reference image)
            ret = time.time() - time_
        else:
            normalizer.fit(reference_image)
            time_ = time.time()
            transformed = normalizer.transform(input_image)  # only report runtime for transform (not considering fit to reference image)
            ret = time.time() - time_

    elif method == "itk":
        time_ = time.time()
        transformed = itk.structure_preserving_color_normalization_filter(  # @TODO: This should include a fit of the reference image as well, right?
        input_image,
        reference_image,
        color_index_suppressed_by_hematoxylin=0,
        color_index_suppressed_by_eosin=1)
        ret = time.time() - time_
    elif method == "torch":
        T = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x*255)
        ])

        torch_normalizer = torchstain.MacenkoNormalizer(backend='torch')
        torch_normalizer.fit(T(reference_image))

        time_ = time.time()
        t_to_transform = T(input_image)
        norm, _, _ = torch_normalizer.normalize(I=t_to_transform, stains=True)
        norm = norm.numpy()
        ret = time.time() - time_
    else:
        raise ValueError
    return transformed, ret


def run_example(new_size=256, method="itk", N_iter=50, figure=True, both=False, normalizer="vahadane"):
    # Fetch images, if we dont have them already.
    input_image_filename = 'Easy1.png'
    input_image_url = 'https://data.kitware.com/api/v1/file/576ad39b8d777f1ecd6702f2/download'
    if not os.path.exists(input_image_filename):
        urlretrieve(input_image_url, input_image_filename)
    reference_image_filename = 'Hard.png'
    reference_image_url = 'https://data.kitware.com/api/v1/file/57718cc48d777f1ecd8a883f/download'
    if not os.path.exists(reference_image_filename):
        urlretrieve(reference_image_url, reference_image_filename)
        
    output_image_filename = 'HardWithEasy1Colors.png'

    # The pixels are RGB triplets of unsigned char.  The images are 2 dimensional.
    PixelType = itk.RGBPixel[itk.UC]
    ImageType = itk.Image[PixelType, 2]

    # Invoke the functional, eager interface for ITK
    input_image = itk.imread(input_image_filename, PixelType)
    reference_image = itk.imread(reference_image_filename, PixelType)
    ImageType = type(input_image)  # get Image type

    # itk image to numpy
    # input_image = itk.array_from_image(input_image)
    input_image = np.asarray(input_image)  # equivalent to the above!
    reference_image = np.asarray(reference_image)

    # resize
    input_image = cv2.resize(input_image, dsize=(new_size, new_size), interpolation=cv2.INTER_LINEAR)
    reference_image = cv2.resize(reference_image, dsize=(new_size, new_size), interpolation=cv2.INTER_LINEAR)

    # numpy to itk image
    input_image = itk.image_from_array(input_image, ttype=(ImageType,))
    reference_image = itk.image_from_array(reference_image, ttype=(ImageType,))

    res = np.zeros(N_iter)
    for i in tqdm(range(N_iter), "Iter:"):
        transformed, ret = normalize(input_image, reference_image, method, both, normalizer)
        res[i] = ret

    print("Runtime results (mu/std):", np.mean(res), np.std(res))

    if figure:
        fig, ax = plt.subplots(1, 3)
        plt.tight_layout()
        ax[0].imshow(np.asarray(input_image))
        ax[1].imshow(np.asarray(reference_image))
        ax[2].imshow(np.asarray(transformed))
        ax[0].set_title("Original")
        ax[1].set_title("Reference")
        ax[2].set_title("Normalized")
        fig.savefig("./subplot.png", dpi=300)


def main(argv):
    # init parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--new_size', metavar='--s', type=int, nargs='?', default=256,
                    help="choose which image size to use.")
    parser.add_argument('--method', metavar='--m', type=str, nargs='?', default="itk",
                    help="choose which method to use.")
    parser.add_argument('--iter', metavar='--i', type=int, nargs='?', default=50,
                    help="number of iterations for runtime estimation.")
    parser.add_argument('--figure', metavar='--f', type=int, nargs='?', default=1,
                    help="whether to make figure.")
    parser.add_argument('--both', metavar='--b', type=int, nargs='?', default=0,
                    help="whether to include fit inside runtime estimation (only relevant for staintools).")
    parser.add_argument('--normalizer', metavar='--n', type=str, nargs='?', default="vahadane",
                    help="choose which normalization method to use for the staintools method (only supported method for itk is 'vahadane').")
    ret = parser.parse_args(); print(ret)

    if ret.method not in ["itk", "staintools", "torch"]:
        raise ValueError("Please, choose between the methods: 'itk', 'staintools', and 'torch'.")
    if ret.normalizer not in ["vahadane", "macenko", "reinhard"]:
        raise ValueError("Please, choose between the methods: 'reinhard', 'macenko', and 'vahadane'.")
    if ret.iter < 1:
        raise ValueError("Please, choose number of iterations to be >= 1.")
    if ret.new_size < 1:
        raise ValueError("Please, choose image size to be >= 1.")
    if ret.figure not in [0, 1]:
        raise ValueError("Please, choose 1 to show figure, else 0.")
    if ret.both not in [0, 1]:
        raise ValueError("Please, choose 1 to include fit inside runtime estimation, else 0.")

    # run
    run_example(*vars(ret).values())


if __name__ == "__main__":
    main(sys.argv[1:])
