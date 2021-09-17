import itk
import argparse
import os
from tqdm import tqdm
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt 
import SimpleITK as sitk


def normalize(input_image, reference_image):
    transformed = itk.structure_preserving_color_normalization_filter(
        input_image,
        reference_image,
        color_index_suppressed_by_hematoxylin=0,
        color_index_suppressed_by_eosin=1
    )
    return transformed


def read(input_image_path, reference_image_path):
    input_image = cv2.imread(input_image_path)[..., ::-1]
    reference_image = cv2.imread(reference_image_path)[..., ::-1]
    return input_image, reference_image


def read_numpy2itk(path):
    ImageType = itk.Image[itk.RGBPixel[itk.UC], 2]
    input_image = cv2.imread(path)[..., ::-1]  # BGR/RGB
    # input_image = np.concatenate([input_image, np.ones(input_image.shape[:2] + (1,))], axis=-1)
    # input_image = itk.image_from_array(input_image, ttype=(ImageType,))
    # return itk.GetImageViewFromArray(input_image)
    return itk.image_from_array(input_image, ttype=(ImageType,))


def subplot(imgs):
    fig, ax = plt.subplots(1, len(imgs))
    for i, img in enumerate(imgs):
        ax[i].imshow(img)
    plt.show()


def run(reference_image_filename, img_path):
    # read reference image only once
    #reference_image = cv2.imread(reference_image_path)[..., ::-1]

    # The pixels are RGB triplets of unsigned char.  The images are 2 dimensional.
    #PixelType = itk.RGBPixel[itk.UC]
    #ImageType = itk.Image[PixelType, 3]
    #ImageType = itk.Image[itk.RGBAPixel[itk.UC], 4]

    # test = cv2.imread(reference_image_filename)
    # print(test.shape, test.dtype, np.unique(test))

    # reference_image_filename = "C:/Users/andrp/workspace/fast-stain-normalization/examples/test/"

    # Invoke the functional, eager interface for ITK
    print(reference_image_filename)
    print(os.path.exists(reference_image_filename))

    # reference_image = itk.imread(reference_image_filename, PixelType)
    # reference_image = itk.imread(reference_image_filename)
    # ImageType = type(reference_image)  # get Image type

    # reader = itk.ImageFileReader(FileName=reference_image_filename)
    #reference_image = reader.GetOutput()

    # reference_image = sitk.ReadImage(reference_image_filename, imageIO="PNGImageIO")

    # reference_image = itk.imread(reference_image_filename, imageIO="PNGImageIO")

    #reference_image = cv2.imread(reference_image_filename)[..., ::-1].astype("uint8")
    # reference_image = reference_image.astype("float32")
    #print(reference_image.shape)
    #reference_image = np.concatenate([reference_image, np.ones(reference_image.shape[:2] + (1,))], axis=-1)
    #print(reference_image.shape)
    #print(reference_image.shape)
    #reference_image = np.array(reference_image, dtype=np.uint16)

    #reference_image = np.concatenate([reference_image, np.ones(reference_image.shape[:2] + (1,))], axis=-1)

    #print(reference_image.shape)

    #print(type(reference_image), reference_image.dtype)

    #reference_image *= 256
    # reference_image = np.expand_dims(reference_image, axis=0)
    print("ref")
    #reference_image = (reference_image / 256).astype("uint8")
    #reference_image = itk.image_from_array(reference_image, ttype=(ImageType,))

    # reference_image = itk.imread(reference_image_filename, itk.F)

    # reference_image = itk.image_from_array(reference_image, ttype=(ImageType,))
    #reference_image = itk.GetImageViewFromArray(reference_image)

    reference_image = read_numpy2itk(reference_image_filename)

    print("type:", type(reference_image))

    print("numpy")
    ret = np.asarray(reference_image)
    print(ret.shape, ret.dtype, type(ret))
    print(np.unique(ret))

    print("---")
    for path in tqdm(os.listdir(img_path), "Image:"):
        # path = "input.jpg"
        print(path)
        input_image_filename = img_path + path
        print(input_image_filename)

        # input_image_filename = reference_image_filename

        # read current image to normalize
        # input_image = itk.imread(input_image_filename, PixelType)
        #reader = itk.ImageFileReader(FileName=input_image_filename)
        #input_image = reader.GetOutput()

        #input_image = cv2.imread(input_image_filename)[..., ::-1]  # .astype(np.uint8)
        #input_image = np.expand_dims(input_image, axis=0)
        #input_image = (input_image / 256).astype("uint8")
        #input_image = itk.image_from_array(input_image, ttype=(ImageType,))

        #input_image = cv2.imread(input_image_filename)[..., ::-1]
        #input_image = np.concatenate([input_image, np.ones(input_image.shape[:2] + (1,))], axis=-1)
        # input_image = itk.image_from_array(input_image, ttype=(ImageType,))
        #input_image = itk.GetImageViewFromArray(input_image)

        input_image = read_numpy2itk(input_image_filename)

        '''
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(input_image)
        ax[1].imshow(reference_image)
        plt.show()
        '''

        subplot([np.asarray(input_image), np.asarray(reference_image)])

        ret = np.asarray(input_image)
        print(ret.shape, ret.dtype, type(ret))
        print(np.unique(ret))

        print()
        print(type(input_image), type(reference_image))

        # normalize
        print("start transform...")
        transformed = normalize(input_image, reference_image)
        print("finished transform")




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
    run(*vars(ret).values())


if __name__ == "__main__":
    main(sys.argv[1:])
