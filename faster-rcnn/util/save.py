import os
import shutil
from enum import Enum

class InputType(Enum):
    DIR = 0
    IMAGE = 1
    VIDEO = 2
    WEBCAM = 3
    OTHERS = 4

image_ext = (
    "jpg", "jepg", "png"
)

video_ext = (
    "mp4", "avi"
)

IMAGE_TYPE = 0


def judge_file_type(path) -> int:
    if path == "0" or path == 0:
        return InputType.WEBCAM
    if os.path.isdir(path):
        return InputType.DIR
    else:
        basename = os.path.basename(path)
        if "." not in basename:
            raise ValueError("Please add extension name for your input!")
        ext = basename.split(".")[-1]
        if ext in image_ext:
            return InputType.IMAGE
        elif ext in video_ext:
            return InputType.VIDEO
        else:
            return InputType.OTHERS


def get_exp_name(listdir : list) -> str:
    """return expi, this is the dir packed training result"""
    next_num = len(listdir)
    listdir = set(listdir)
    while True:
        exp_name = "exp" + str(next_num)
        if exp_name in listdir:
            next_num += 1
        else:
            break
    if exp_name == "exp0":
        exp_name = "exp"
    return exp_name

def find_best_model(weights_dir : str) -> None:
    """find best model (val is the lowest) in weight dir then remove others and rename best model as best.pth"""
    print(weights_dir)
    assert os.path.exists(weights_dir)
    best_score = float("inf")
    best_weight = ""
    for weight_name in os.listdir(weights_dir):
        if "_" not in weight_name:
            continue
        _, val_loss = weight_name.split("_")
        val_loss = float(val_loss)
        if val_loss < best_score:
            best_score = val_loss
            best_weight = weight_name
    
    for weight_name in os.listdir(weights_dir):
        if "_" not in weight_name:
            continue
        weight_path = os.path.join(weights_dir, weight_name)
        if weight_name == best_weight:
            shutil.move(weight_path, os.path.join(weights_dir, "best.pth"))
        else:
            os.remove(weight_path)