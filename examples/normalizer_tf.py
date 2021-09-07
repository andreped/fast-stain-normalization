# necessary for openslide on Windows
import os
os.environ['PATH'] = "C:/Users/andrp/Downloads/openslide-win64-20171122/openslide-win64-20171122/bin" + ";" + os.environ['PATH']

import sys
import argparse
from Fast_WSI_Color_Norm.Run_ColorNorm import run_colornorm
import tensorflow as tf



def run(reference_image_filename, img_path):
	# run_stainsep(reference_image_filename, nstains, lamb)

	os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

	config = tf.ConfigProto(log_device_placement=False, gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=1))
	background_correction = True	
	lamb = 0.01
	nstains = 2
	output_direc = "./"
	level = 0
	run_colornorm(
		img_path,
		reference_image_filename,
		nstains,
		lamb,
		output_direc,
		level,
		background_correction,
		config=config,
	)



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
