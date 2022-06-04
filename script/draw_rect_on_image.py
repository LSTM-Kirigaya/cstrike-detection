import cv2 as cv
import os
import sys
sys.path.append('.')

from util.cv_util import show_image
from util.yolo_util import yolo_label_to_pts, random_color_by_cls, cls_to_name
from util.data_util import read_meta

IMAGE_ROOT = "./datasets/images"
META_PATH = "./meta.json"
CLASS_TXT = "./datasets/labels/classes.txt"
SHOW_IMAGE_INDEX = 1

image_file = os.listdir(IMAGE_ROOT)[SHOW_IMAGE_INDEX]
image_path = os.path.join(IMAGE_ROOT, image_file)
image_index = image_file.split(".")[0]
image = cv.imread(image_path)
colormap = random_color_by_cls()
clsmap = cls_to_name(CLASS_TXT)
yolo_labels = read_meta(META_PATH)[image_index]

for yl in yolo_labels:
    cls_lab = yl[-1]
    x_min, x_max, y_min, y_max = yolo_label_to_pts(yl[:-1], image.shape)
    image = cv.rectangle(
        image, 
        (x_min, y_min), 
        (x_max, y_max), 
        colormap[cls_lab], 
        thickness=2
    )
    
    print(colormap[cls_lab])
    image = cv.putText(
        image, 
        text=clsmap[cls_lab], 
        org=(x_min, y_min - 10), 
        fontFace=cv.FONT_HERSHEY_SIMPLEX,
        fontScale=1, 
        color=colormap[cls_lab], 
        thickness=2
    )


    
show_image(image)