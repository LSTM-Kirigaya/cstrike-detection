import cv2 as cv
import typing
import numpy as np

def yolo_label_to_pts(yolo_label : typing.List[float], image_shape : typing.Tuple[int]) -> typing.Tuple[int]:
    """transform yolo label to [x_min, y_min, x_max, y_max]"""
    ih, iw = image_shape[:2]
    x = iw * yolo_label[0]
    y = ih * yolo_label[1]
    bw = iw * yolo_label[2]
    bh = ih * yolo_label[3]

    x_min = int(x - bw / 2)
    x_max = int(x + bw / 2)
    y_min = int(y - bh / 2)
    y_max = int(y + bh / 2)

    return (x_min, x_max, y_min, y_max)

def cls_to_name(classtxt : str) -> dict:
    """map from cls index to its real name"""
    result = {}
    for i, line in enumerate(open(classtxt, "r", encoding="utf-8")):
        result[i] = line.strip()
    return result

def random_color_by_cls(cls_num : int = None) -> dict:
    """generate map from class to color"""
    if cls_num:
        colormap = {i : (
            np.random.randint(low=0, high=256),
            np.random.randint(low=0, high=256),
            np.random.randint(low=0, high=256)
        ) for i in range(cls_num) }
    else:
        colormap = {
            0 : ( 89, 221,  64),         # C
            1 : (253, 111,  54),         # dead
            2 : ( 92, 155, 255),         # T
            3 : (128, 30, 255),          # weapon
        }
    return colormap