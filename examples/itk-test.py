from urllib.request import urlretrieve
import os
import itk
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import staintools


def normalize(input_image, reference_image, method):
    if method == "staintools":
        normalizer = staintools.StainNormalizer(method='vahadane')
        normalizer.fit(reference_image)
        time_ = time.clock()
        transformed = normalizer.transform(input_image)
    elif method == "itk":
        time_ = time.clock()
        transformed = itk.structure_preserving_color_normalization_filter(
        input_image,
        reference_image,
        color_index_suppressed_by_hematoxylin=0,
        color_index_suppressed_by_eosin=1)
    else:
        raise ValueError
    return transformed, time.clock() - time_


# Fetch input images, if we don have them already.
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

ImageType = type(input_image)

# resize images to fixed size
new_size = 256
N_iter = 50
method = "itk"  # itk, staintools

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

print("--")

res = np.zeros(N_iter)
for i in range(N_iter):
    time_ = time.clock()
    transformed, ret = normalize(input_image, reference_image, method)
    res[i] = ret

    # print(res)

print("Runtime results (mu/std):", np.mean(res), np.std(res))

fig, ax = plt.subplots(1, 3)
plt.tight_layout()
ax[0].imshow(np.asarray(input_image))
ax[1].imshow(np.asarray(reference_image))
ax[2].imshow(np.asarray(transformed))
ax[0].set_title("Original")
ax[1].set_title("Reference")
ax[2].set_title("Normalized")
fig.savefig("./subplot.png", dpi=300)
