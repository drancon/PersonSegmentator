import os
import argparse
import numpy as np
import cv2
from tqdm import tqdm

from person_segmentator import PersonSegmentator

def parse_args():
    parser = argparse.ArgumentParser(
        description='run person segmentator on example images')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--data_dir', default='demo', help='a directory containing input images')
    parser.add_argument('--save_root', default=None, help='a root directory for saving the masked people images')
    args = parser.parse_args()

    return args

def main():
    # get the arguments
    args = parse_args()

    # initialize a segmentator
    segmentator = PersonSegmentator(cfg_path=args.config, ckpt_path=args.checkpoint, gpu_idx=0)

    # list the names of images in the data_dir
    files = [ name for name in os.listdir(args.data_dir) if '.jpg' in name[-4:].lower() ]
    # load images from the data_dir
    for file in tqdm(files):
        ## load the image
        img = cv2.imread(os.path.join(args.data_dir, file))
        ## set the save_dir for curent image
        if args.save_root is not None:
            save_dir = os.path.join(args.save_root, file[:-4])
        else:
            save_dir = None
        ## run the segmentator on the image
        results = segmentator.detect(img, ths=args.threshold, save_dir=save_dir)

if __name__ == '__main__':
    main()