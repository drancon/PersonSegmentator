import os
import warnings
import json
import cv2
import numpy as np

import mmcv
import torch
from mmcv import Config
from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.runner import load_checkpoint, wrap_fp16_model

from mmseg.apis import inference_segmentor
from mmseg.models import build_segmentor

class PersonSegmentator():
    """ Segmentator based on SegFormer
    Initialize the person segmentation model based on SegFormer
    and output person segmentation masks
    
    Args:
        cfg_path (str): path to the configuration file of object segmentation model, the configuration follows the format of backend packages(mmcv and mmseg)
        ckpt_path (str): path to the checkpoint(traiend weights) of the model
        is_fuse_conv_bn (bool): a bool variable to determine if to fuse conv and bn, this will slightly increase the inference speed
        gpu_idx (int): gpu index for running this model
    """
    def __init__(self, cfg_path, ckpt_path, gpu_idx=0):
        # check if input variables are valid
        if not isinstance(gpu_idx, int):
            raise ValueError('The input variable "gpu_idx" must be an integer.')

        # store the input variables into class variables
        self.cfg = Config.fromfile(cfg_path)
        self.ckpt_path = ckpt_path
        self.device = torch.device("cuda:%d"%gpu_idx if torch.cuda.is_available() else "cpu")
        # initialize the detector model
        self._init_model()

    def _init_model(self):
        """ Initialize the segmentation model and load checkpoint file """
        # build detector
        self.cfg.model.train_cfg = None
        model = build_segmentor(self.cfg.model)
        # convert the datatype of weights in the detector into fp16
        fp16_cfg = self.cfg.get('fp16', None)
        if fp16_cfg is not None:
            wrap_fp16_model(model)
        # load the saved model weights
        checkpoint = load_checkpoint(model, self.ckpt_path)
        # sync_batchnorm used only in distributed executions
        model = revert_sync_batchnorm(model)

        # old versions did not save class info in checkpoints, 
        # this walkaround is for backward compatibility
        model.CLASSES = checkpoint['meta']['CLASSES']
        # save the config in the model for convenience
        model.cfg = self.cfg
        # transfer the model into gpu
        model.to(self.device)
        # set the model as evaluation mode
        model.eval()

        # store the initialized model into class variable
        self.model = model

    def detect(self, img, min_size=(256,512), save_dir=None, save_ext='png'):
        """ make person in the input image and save the result image

        Args:
            img (numpy array, shape:[H, W, 3]): an input image
            min_size (tuple, (W_min, H_min)): the minimum size of input image, small input image will be resized to a larger image
            save_dir (str): a directory for saving the result images
            save_ext (str): the file extension of result images to be saved

        Returns:
            cls_img (list): an image(numpy array, shape:[H, W, 3]) of people masked in the input image
        """
        # check if input variables are valid
        if save_dir is not None:
            if not isinstance(save_dir, str):
                raise ValueError('The input variable "save_dir" should be in the type of "str" or "None".')
        if save_ext.lower() not in ['png', 'jpg', 'jpeg', 'bmp']:
            raise ValueError('The input variable "save_ext" should be either one of "png", "jpg", "jpeg", or "bmp".')

        # get the size of input image
        H, W, _ = img.shape
        resize_img = False
        # check if the minimum size input is valid
        if min_size[0] > W or min_size[1] > H:
            resize_img = True

        # detect people in the input image
        img_ori = img.copy()
        x0, y0, x1, y1 = 0, 0, W-1, H-1
        if resize_img:
            # resize the input image larger
            img_ = cv2.resize(img, (min_size[0], min_size[1]))
            # set the indices of center box to store input image
            x0 = int((2048 - min_size[0])/2)
            x1 = x0 + min_size[0]
            y0 = int((1024 - min_size[1])/2)
            y1 = y0 + min_size[1]
            # pad the input image with zero
            img = np.zeros([1024, 2048, 3])
            img[y0:y1, x0:x1] = img_
        results = inference_segmentor(self.model, img)
        
        # get the segmentation result of input image
        result = results[0]

        # initialize a list to store person image mask
        # generate mask images for person
        ## 11: pedestrian, 12: rider in cityscapes dataset
        cls_img = (result==11) + (result==12)
        cls_img = 255*cls_img
        if resize_img:
            cls_img = cls_img[y0:y1, x0:x1]
            fx = W / min_size[0]
            fy = H / min_size[1]
            cls_img = cv2.resize(cls_img.astype(np.uint8), None, fx=fx, fy=fy, interpolation=cv2.INTER_AREA)
        # generate an overwrap image
        fuse_img = 0.7*img_ori + 0.3*cls_img[:,:,None]*np.array([255,0,0])

        # save the result images
        if save_dir is not None:
            # if the directory does not exist, create the directory
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            # save the images into the directory
            cv2.imwrite(os.path.join(save_dir, "mask.%s"%(save_ext)), cls_img)
            cv2.imwrite(os.path.join(save_dir, "mask_fused.%s"%(save_ext)), fuse_img)
        return cls_img