import numpy as np
import os
import shutil

DATA_ROOT = "data"
LABEL_ROOT = "label"

TARGET_TRAIN_DIR = "./mobilenet-v2/Tensorflow/workspace/images/train"
TARGET_TEST_DIR = "./mobilenet-v2/Tensorflow/workspace/images/test"

all_images = os.listdir(DATA_ROOT)
indice = np.arange(len(all_images))

for i in indice[:-10]:
    name = all_images[i].rstrip(".png")
    image_path = os.path.join(DATA_ROOT, name + ".png")
    label_path = os.path.join(LABEL_ROOT, name + ".xml")
    shutil.copy(image_path, os.path.join(TARGET_TRAIN_DIR, name + ".png"))
    shutil.copy(label_path, os.path.join(TARGET_TRAIN_DIR, name + ".xml"))

for i in indice[-10:]:
    name = all_images[i].rstrip(".png")
    image_path = os.path.join(DATA_ROOT, name + ".png")
    label_path = os.path.join(LABEL_ROOT, name + ".xml")
    shutil.copy(image_path, os.path.join(TARGET_TEST_DIR, name + ".png"))
    shutil.copy(label_path, os.path.join(TARGET_TEST_DIR, name + ".xml"))