import argparse
import torch
from torchvision import transforms
import torchstain
import cv2
import sys
import matplotlib.pyplot as plt
import warnings
import os
from datetime import datetime, date
from tqdm import tqdm
from math import ceil
import time
from functools import partial
import multiprocessing as mp


# mute all warnings
warnings.filterwarnings("ignore")

# enable efficient torch compute (only relevant for GPU)
torch.backends.cudnn.benchmark = True


def generator(x, batch_size, N):
    for idx in range(int(ceil(N / batch_size))):
        yield x[idx * batch_size:(idx + 1) * batch_size]


def get_time():
    curr_date = "".join(date.today().strftime("%d/%m").split("/")) + date.today().strftime("%Y")[2:]
    curr_time = "".join(str(datetime.now()).split(" ")[1].split(".")[0].split(":"))
    return curr_date, curr_time


def subplot(imgs):
    fig, ax = plt.subplots(1, len(imgs))
    for i, img in enumerate(imgs):
        ax[i].imshow(img)
    plt.show()


def single_transform(img_path, device, save_path, torch_normalizer, path, backend):
    T = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x*255)
    ])

    if path.split(".")[-1].lower() not in ["png", "jpg", "jpeg", "tif", "tiff"]:
        print("\nImage format not supported:", img_path + path)
        exit()
    to_transform = cv2.cvtColor(cv2.imread(img_path + path), cv2.COLOR_BGR2RGB)
    t_to_transform = T(to_transform)

    if backend == "torch":
        t_to_transform = t_to_transform.to(device)

    norm, _, _ = torch_normalizer.normalize(I=t_to_transform, stains=True)

    if (backend == "torch") and (device != "cpu"):
        norm = norm.cpu().numpy().astype("uint8")

    cv2.imwrite(save_path + path, cv2.cvtColor(norm, cv2.COLOR_RGB2BGR))


def run(reference_image_filename, img_path, out_path, cpu, parallel, workers, backend):
    if torch.cuda.is_available() and (cpu == 0):
        device = torch.device("cuda:0")  # @FIXME: uses first CUDA-compatible GPU, add support for choosing which GPU relevant for multi-GPU setups
    else:
        device = torch.device("cpu")

    target = cv2.cvtColor(cv2.imread(reference_image_filename), cv2.COLOR_BGR2RGB)
    to_transform = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

    T = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x*255)
    ])

    torch_normalizer = torchstain.MacenkoNormalizer(backend=backend)
    torch_normalizer.fit(T(target).to(device))

    t_to_transform = T(to_transform).to(device)
    norm, _, _ = torch_normalizer.normalize(I=t_to_transform, stains=True)
    if (backend == "torch") and (device != "cpu"):
        norm = norm.cpu().numpy().astype("uint8")

    curr_date, curr_time = get_time()
    save_path = out_path + "output_normalization_" + curr_date + "_" + curr_time + "/"
    os.makedirs(save_path, exist_ok=True)
    cv2.imwrite(save_path + img_path.split("/")[-1], cv2.cvtColor(norm, cv2.COLOR_RGB2BGR))


def run_batch(reference_image_filename, img_path, out_path, cpu, parallel, workers, backend):
    if torch.cuda.is_available() and (cpu == 0):
        device = torch.device("cuda:0")  # @FIXME: uses first CUDA-compatible GPU, add support for choosing which GPU relevant for multi-GPU setups
    else:
        device = torch.device("cpu")

    if reference_image_filename.split(".")[-1].lower() not in ["png", "jpg", "jpeg", "tif", "tiff"]:
        raise ValueError("\nImage format not supported:", reference_image_filename)
    target = cv2.cvtColor(cv2.imread(reference_image_filename), cv2.COLOR_BGR2RGB)

    T = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x*255)
    ])

    torch_normalizer = torchstain.MacenkoNormalizer(backend=backend)
    torch_normalizer.fit(T(target).to(device))

    curr_date, curr_time = get_time()
    save_path = out_path + "output_normalization_" + curr_date + "_" + curr_time + "/" + img_path.split("/")[-2] + "/"
    os.makedirs(save_path, exist_ok=True)

    # sequential (batch_size = 1)
    for path in tqdm(os.listdir(img_path), "Images:"):
        if path.split(".")[-1].lower() not in ["png", "jpg", "jpeg", "tif", "tiff"]:
            print("\nImage format not supported:", img_path + path)
            continue
        to_transform = cv2.cvtColor(cv2.imread(img_path + path), cv2.COLOR_BGR2RGB)

        t_to_transform = T(to_transform).to(device)
        norm, _, _ = torch_normalizer.normalize(I=t_to_transform, stains=True)
        if (backend == "torch") and (device != "cpu"):
            norm = norm.cpu().numpy().astype("uint8")

        cv2.imwrite(save_path + path, cv2.cvtColor(norm, cv2.COLOR_RGB2BGR))


def run_batch_parallel(reference_image_filename, img_path, out_path, cpu, parallel, workers, backend):
    if torch.cuda.is_available() and (cpu == 0):
        device = torch.device("cuda:0")  # @FIXME: uses first CUDA-compatible GPU, add support for choosing which GPU relevant for multi-GPU setups
    else:
        device = torch.device("cpu")

    if reference_image_filename.split(".")[-1].lower() not in ["png", "jpg", "jpeg", "tif", "tiff"]:
        raise ValueError("\nImage format not supported:", reference_image_filename)
    target = cv2.cvtColor(cv2.imread(reference_image_filename), cv2.COLOR_BGR2RGB)

    T = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x*255)
    ])

    torch_normalizer = torchstain.MacenkoNormalizer(backend=backend)
    torch_normalizer.fit(T(target).to(device))

    curr_date, curr_time = get_time()
    save_path = out_path + "output_normalization_" + curr_date + "_" + curr_time + "/" + img_path.split("/")[-2] + "/"
    os.makedirs(save_path, exist_ok=True)
    
    tmp = os.listdir(img_path)
    with mp.Pool(processes=workers) as p:
        ret = list(tqdm(p.imap_unordered(partial(single_transform, img_path, device, save_path, torch_normalizer, backend), tmp), total=len(tmp)))


def main():
    # init parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref', metavar='--r', type=str, nargs='?',
                    help="set which reference image to use.")
    parser.add_argument('--img', metavar='--i', type=str, nargs='?',
                    help="set path to which folder containing the images to normalize.")
    parser.add_argument('--out', metavar='--o', type=str, nargs='?', default="./",
                    help="set path to store the output.")
    parser.add_argument('--cpu', metavar='--c', type=int, nargs='?', default=1,
                    help="force computations to use CPU (GPU is disabled by default).")
    parser.add_argument('--mp', metavar='--c', type=int, nargs='?', default=0,
                    help="enable multiprocessing.")
    parser.add_argument('--wk', metavar='--c', type=int, nargs='?', default=1,
                    help="set number of workers relevant for multiprocessing.")
    parser.add_argument('--backend', metavar='--b', type=int, nargs='?', default="torch",
                    help="set which backend to use for normalization. Torch used as default.")
    ret = parser.parse_args(sys.argv[1:]); print(ret)

    if ret.ref is None:
        raise ValueError("Please, set path to the reference image you wish to use.")
    if ret.img is None:
        raise ValueError("Please, set path to the folder containing the images to normalize.")
    if not os.path.exists(ret.ref):
        raise ValueError("Reference image provided does not exist!")
    if not os.path.exists(ret.img):
        raise ValueError("Image path provided does not exist!")
    if not ret.cpu in [0, 1]:
        raise ValueError("Please, choose '1' to force CPU compute, or '0' to not.")
    if not ret.mp in [0, 1]:
        raise ValueError("Please, choose '1' to enable multiprocessing, or '0' to not.")
    if (ret.mp == 1) and (ret.wk < 1):
        raise ValueError("Number of workers has to be >= 1.")
    if (ret.mp == 1) and (ret.cpu == 0):
        raise NotImplementedError("Does not support multiprocessing on GPU.")
    if ret.backend not in ["torch", "tensorflow"]:
        raise ValueError("Please, choose either 'torch' or 'tensorflow' as backend.")

    # cap cores
    max_ = mp.cpu_count()
    workers = max_ if ret.wk > max_ else ret.wk

    # fix paths
    ret.img = ret.img.replace("\\", "/")
    ret.ref = ret.ref.replace("\\", "/")
    ret.out = ret.out.replace("\\", "/")

    # run
    if os.path.isdir(ret.img):
        if not ret.img.endswith("/"):
            ret.img += "/"
        if ret.mp == 0:
            run_batch(*vars(ret).values())
        else:
            run_batch_parallel(*vars(ret).values())
    else:
        run(*vars(ret).values())


if __name__ == '__main__':
    main()
