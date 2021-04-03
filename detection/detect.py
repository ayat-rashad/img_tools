import sys

sys.path.append("../")

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from PIL import Image
import scrapbook as sb
import torch
import torchvision

from .dataset import DetectionDataset
from .model import DetectionLearner, get_pretrained_maskrcnn

from skimage import io
import numpy as np

from img_tools import preprocess as pp

import logging

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using torch device: {device}")

# get pretrained model
detector = DetectionLearner(
    model=get_pretrained_maskrcnn(), 
    device=device,
    labels=coco_labels()[1:],
)


def apply_detection_dir(data_root):
    def is_valid_img(img):
        # TODO : add serious checks
        return img.endswith(('.jpg', '.jpeg'))
    
    join = os.path.join
    imgs = [join(data_root, im) for im in os.listdir(data_root) if is_valid_img(im)]

    if len(imgs) == 0:
        logger.info('empty folder..')
        return 
    
    for im_path in imgs:
        try:
            im = io.imread(im_path)

            detections = detector.predict(im_path, threshold=.3)
            bboxs = [[bbox.top, bbox.left, bbox.bottom,bbox.right] for bbox in detections['det_bboxes']]
            im2, main_bbox = pp.crop_main_bbox(im, bboxs)

            im_path2, im2_name = os.path.dirname(im_path), os.path.basename(im_path)
            im_path2 = join(im_path2, 'od_' + im2_name)
            io.imsave(im_path2, im2)
        except Exception as e:
            logger.error(e)
    
            

def apply_detection_recur(data_root):
    join = os.path.join
     
    # apply detection to root directory
    apply_detection_dir(data_root)
    
    # loop recursively
    dirs = [join(data_root, d) for d in os.listdir(data_root) if os.path.isdir(join(data_root, d))]

    for d in dirs:
        logger.info("processing directory: %s" %d)
        apply_detection_recur(d)
