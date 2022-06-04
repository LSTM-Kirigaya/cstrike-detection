import json
import os
import numpy as np

ROOT_DIR = "datasets"
IMAGE_DIR = os.path.join(ROOT_DIR, "images")
LABEL_DIR = os.path.join(ROOT_DIR, "labels")
EXCLUDE_FILE = ["classes.txt"]
TEST_NUM = 20

all_images = os.listdir(IMAGE_DIR)
indice = np.arange(len(all_images))
print(len(all_images))
np.random.shuffle(indice)
train_image_fp = open(os.path.join(ROOT_DIR, "traindata.txt"), "w", encoding="utf-8")
test_image_fp = open(os.path.join(ROOT_DIR, "testdata.txt"), "w", encoding="utf-8")

for i in indice[:-TEST_NUM]:
    abspath = os.path.abspath(os.path.join(IMAGE_DIR, all_images[i]))
    train_image_fp.write(abspath)
    train_image_fp.write("\n")

for i in indice[-TEST_NUM:]:
    abspath = os.path.abspath(os.path.join(IMAGE_DIR, all_images[i]))
    test_image_fp.write(abspath)
    test_image_fp.write("\n")

train_image_fp.close()
test_image_fp.close()