import torch
from torch.utils.data import Dataset
import typing
import numpy as np
import os
import cv2

class CS(Dataset):
    def __init__(self, data_meta : list, image_root_dir : str, ):
        super().__init__()
        self.meta_dict = {}
        for meta in data_meta:
            image_id = meta[0]
            bbox_labels = meta[1]
            self.meta_dict[image_id] = {
                "bbox" : [],
                "label" : [],
                "area" : [],
            }
            image_path = os.path.join(image_root_dir, image_id + ".png")
            image = cv2.imread(image_path)
            h, w = image.shape[:2]
            for bbox_label in bbox_labels:
                x1 = int(w * bbox_label[1])
                y1 = int(h * bbox_label[2])
                x2 = int(w * bbox_label[3])
                y2 = int(h * bbox_label[4])
                self.meta_dict[image_id]["bbox"].append([x1, y1, x2, y2])
                self.meta_dict[image_id]["label"].append(int(bbox_label[0]))
                self.meta_dict[image_id]["area"].append(float((x2 - x1) * (y2 - y1)))
        
        self.image_root_dir = image_root_dir
        self.image_ids = list(self.meta_dict.keys())

    def __getitem__(self, index : int) -> tuple:
        image_id = self.image_ids[index]
        image_path = os.path.join(self.image_root_dir, image_id + ".png")
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype("float32")
        image /= 255.

        boxes = self.meta_dict[image_id]["bbox"]
        area = self.meta_dict[image_id]["area"]
        labels = self.meta_dict[image_id]["label"]
        iscrowd = torch.zeros((len(labels),), dtype=torch.uint8)

        target = {}
        target["boxes"] = torch.FloatTensor(boxes)
        if len(target["boxes"]) == 0:
            target["boxes"] = torch.zeros((0, 4))
        target["labels"] = torch.LongTensor(labels)
        target["image_id"] = torch.LongTensor([index])
        target["area"] = torch.FloatTensor(area)
        target["iscrowd"] = iscrowd

        image = torch.FloatTensor(image).permute((2, 0, 1))

        return image, target, image_id
    
    def __len__(self):
        return len(self.image_ids)
    
