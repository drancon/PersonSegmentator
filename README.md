# PersonSegmentator
Find people in images and generate masked people images.  
The segmentation model is made based on SegFormer(https://github.com/NVlabs/SegFormer).  
This repository uses SegFormer semantic segmentation module with MIT-B5 backbone provided by MMSegmentation(https://github.com/open-mmlab/mmsegmentation/tree/master/configs/segformer).
# Pretrained Weights
- download *.pth file from https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b5_8x1_1024x1024_160k_cityscapes/segformer_mit-b5_8x1_1024x1024_160k_cityscapes_20211206_072934-87a052ec.pth
- move the downloaded *.pth file to ckpts/segformer_mit-b5_8x1_1024x1024_160k_cityscapes.pth