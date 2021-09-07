import argparse
import torch
from torchvision import transforms
import torchstain
import cv2
import sys
import matplotlib.pyplot as plt
import warnings
import os


# mute all warnings
warnings.filterwarnings("ignore")


def subplot(imgs):
    fig, ax = plt.subplots(1, len(imgs))
    for i, img in enumerate(imgs):
        ax[i].imshow(img)
    plt.show()


def run(reference_image_filename, img_path):

    target = cv2.cvtColor(cv2.imread(reference_image_filename), cv2.COLOR_BGR2RGB)
    to_transform = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

    T = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x*255)
    ])

    torch_normalizer = torchstain.MacenkoNormalizer(backend='torch')
    torch_normalizer.fit(T(target))

    t_to_transform = T(to_transform)
    norm, H, E = torch_normalizer.normalize(I=t_to_transform, stains=True)

    return norm


def run_batch(reference_image_filename, img_path):

    target = cv2.cvtColor(cv2.imread(reference_image_filename), cv2.COLOR_BGR2RGB)

    T = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x*255)
    ])

    torch_normalizer = torchstain.MacenkoNormalizer(backend='torch')
    torch_normalizer.fit(T(target))

    for path in os.listdir(img_path):
        to_transform = cv2.cvtColor(cv2.imread(img_path + path), cv2.COLOR_BGR2RGB)

        t_to_transform = T(to_transform)
        norm, _, _ = torch_normalizer.normalize(I=t_to_transform, stains=True)


def main(argv):
    # init parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref', metavar='--r', type=str, nargs='?',
                    help="set which reference image to use.")
    parser.add_argument('--img', metavar='--i', type=str, nargs='?',
                    help="set path to which folder containing the images to normalize.")
    ret = parser.parse_args(); print(ret)

    if ret.ref is None:
        raise ValueError("Please, set path to the reference image you wish to use.")

    if ret.img is None:
        raise ValueError("Please, set path to the folder containing the images to normalize.")

    # run
    if os.path.exists(ret.img):
        if os.path.isdir(ret.img):
            run_batch(*vars(ret).values())
        else:
            run(*vars(ret).values())
    else:
        raise ValueError


if __name__ == "__main__":
    main(sys.argv[1:])
