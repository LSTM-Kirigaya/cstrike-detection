import os
import shutil
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:
    plt.style.use("gadfly")
except:
    pass

MAKE_META_SCRIPT = "make_meta.py"
IMAGE_DIR = "datasets/images"
LABEL_DIR = "datasets/labels"
TESTDATATXT = "datasets/testdata.txt"
PYTHON_INTERPRETER = "python" if os.name == 'nt' else "python3"
TEMP_NAME = "temp"
RES_NAME = "result"

TEST_MODEL = ["faster-rcnn", "yolov5"]
CLASS_NAME = ["C", "dead", "T", "weapon"]

assert os.path.exists(IMAGE_DIR)
assert os.path.exists(LABEL_DIR)
if not os.path.exists(TESTDATATXT):
    os.system("{} {}".format(PYTHON_INTERPRETER, os.path.join("script", MAKE_META_SCRIPT)))

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolov5-weights', help="path of yolov5 path")
    parser.add_argument('--faster-rcnn-weights', help="path of faster-rcnn-weights")

    return parser.parse_args()

def make_test_image_dir():
    for line in open(TESTDATATXT, "r", encoding="utf-8"):
        image_path = line.strip()
        image_name = os.path.basename(image_path)
        shutil.copy(image_path, os.path.join(TEMP_NAME, image_name))

def run_yolo(weights):
    cmd = "{} yolov5/detect.py --source temp --project result --name yolov5 --save-txt --save-conf --weights {} --device 0 --img 640 --nosave".format(
        PYTHON_INTERPRETER, weights
    )
    os.system(cmd)

def run_faster_rcnn(weights):
    cmd = "{} faster-rcnn/detect.py --source temp --project result --name faster-rcnn --save-txt --save-conf --weights {} --device cuda --nosave".format(
        PYTHON_INTERPRETER, weights
    )
    os.system(cmd)

def make_pred_labels(model_name : str) -> pd.DataFrame:
    pred = {
        "image_id" : [],
        "label" : [],
        "confidence" : [],
        "x" : [],
        "y" : [],
        "w" : [],
        "h" : []
    }

    # acquire pred
    txt_path = os.path.join(os.path.join(RES_NAME, model_name), "labels")
    for file in os.listdir(txt_path):
        label_path = os.path.join(txt_path, file)
        image_id = file.split(".")[0]
        for line in open(label_path, "r", encoding="utf-8"):
            datas = line.split()
            # datas : label x y w h conf
            pred["label"].append(int(datas[0]))
            pred["image_id"].append(image_id)
            pred["x"].append(float(datas[1]))
            pred["y"].append(float(datas[2]))
            pred["w"].append(float(datas[3]))
            pred["h"].append(float(datas[4]))
            pred["confidence"].append(float(datas[5]))

    return pd.DataFrame(pred)

def make_gt_label() -> pd.DataFrame:
    gt = {
        "image_id" : [],
        "label" : [],
        "x" : [],
        "y" : [],
        "w" : [],
        "h" : [],
    }

    for file in os.listdir(TEMP_NAME):
        image_id = os.path.basename(file).split(".")[0]
        gt_label_path = os.path.join(LABEL_DIR, image_id + ".txt")
        for line in open(gt_label_path, "r", encoding="utf-8"):
            datas = line.split()
            gt["image_id"].append(image_id)
            gt["label"].append(int(datas[0]))
            gt["x"].append(float(datas[1]))
            gt["y"].append(float(datas[2]))
            gt["w"].append(float(datas[3]))
            gt["h"].append(float(datas[4]))


    return pd.DataFrame(gt)

def xywh2xyxy(x, y, w, h):
    return (
        x - w / 2,
        y - h / 2,
        x + w / 2,
        y + h / 2
    )

def IoU(box1, box2) -> float:
    """box must be (x_min, y_min, x_max, y_max) style"""
    x1 = max(box1[0], box2[0])   
    x2 = min(box1[2], box2[2])   
    y1 = max(box1[1], box2[1])  
    y2 = min(box1[3], box2[3])  
 
    overlap = max(0., x2 - x1) * max(0., y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - overlap
 
    return overlap / union


def AP(points : np.ndarray) -> float:
    """cal area under PR curve"""
    area = 0
    for i in range(len(points) - 1):
        width = points[i + 1][0] - points[i][0]
        height = (points[i + 1][1] + points[i][1]) / 2
        area += width * height
    return area

def cal_ap_one_class(pred : pd.DataFrame, gt : pd.DataFrame, iou_thred : float = 0.5) -> np.ndarray:
    """cal ap of a class return numpy like List[(recall, precision)]"""
    # cal TP or FP of each line in pred
    ap_view = {
        "image_id" : [],
        "confidence" : [],
        "TP" : [],          # 1 is tp and 0 is fp
        "box_loss" : [],    # if TP == 0 then this item is -1
    }

    # loop each image
    for image_id, data in pred.groupby("image_id"):
        data : pd.DataFrame

        # align each line to a box or None
        image_id_gt = gt[gt["image_id"] == image_id]
        # recurse each line in this image
        for i in range(len(data)):
            pred_view = data.iloc[i]
            tp = 0
            box_loss = -1
            max_iou = -1
            pred_box = xywh2xyxy(pred_view.x, pred_view.y, pred_view.w, pred_view.h)
            for j in range(len(image_id_gt)):
                gt_view = image_id_gt.iloc[j]
                gt_box = xywh2xyxy(gt_view.x, gt_view.y, gt_view.w, gt_view.h)
                iou = IoU(pred_box, gt_box)
                if iou >= iou_thred:
                    if max_iou == -1 or iou >= max_iou:
                        max_iou = iou
                        tp = 1
                        box_loss = np.linalg.norm(np.array(pred_box) - np.array(gt_box))
            ap_view["image_id"].append(pred_view.image_id)
            ap_view["confidence"].append(pred_view.confidence)
            ap_view["TP"].append(tp)
            ap_view["box_loss"].append(box_loss)

    ap_df = pd.DataFrame(ap_view)
    ap_df = ap_df.sort_values(by="confidence", ascending=False)
    points = np.zeros((len(ap_df), 2))
    positive_num = len(gt)
    cur_tp = 0

    all_box_loss = []

    for i in range(len(ap_df)):
        view = ap_df.iloc[i]
        cur_tp += view.TP
        if view.box_loss > 0:
            all_box_loss.append(view.box_loss)
        precision = cur_tp / (i + 1)
        recall = cur_tp / positive_num
        points[i] = (recall, precision)
    
    ap_value = AP(points)
    box_loss = np.mean(all_box_loss)
    return ap_value, box_loss, points

if __name__ == "__main__":
    if os.path.exists(TEMP_NAME):
        shutil.rmtree(TEMP_NAME)
    os.mkdir(TEMP_NAME)
    if os.path.exists(RES_NAME):
        shutil.rmtree(RES_NAME)
    os.mkdir(RES_NAME)

    args = get_args()
    make_test_image_dir()
    
    run_yolo(args.yolov5_weights)
    run_faster_rcnn(args.faster_rcnn_weights)

    gt = make_gt_label()
    plt.figure(dpi=120)

    for i in range(len(TEST_MODEL)):
        plt.subplot(1, len(TEST_MODEL), i + 1)
        plt.title("P-R/{}".format(TEST_MODEL[i].upper()))
        label_nums = []
        ap_values = []
        box_losses = []
        pred = make_pred_labels(TEST_MODEL[i])
        for label, data in gt.groupby("label"):
            label_pred = pred[pred.label == label]
            label_gt = data
            label_nums.append(len(data))
            ap_value, box_loss, points = cal_ap_one_class(label_pred, label_gt, iou_thred=0.5)
            ap_values.append(ap_value)
            box_losses.append(box_loss)
            plt.plot(points[..., 0], points[..., 1], '-o', label=CLASS_NAME[label])

        weights = np.array(label_nums)
        weights = weights / weights.sum()
        print("[LOG] {} mAP 0.5 {}, mAP 0.5 (weights) {}, box loss {}".format(
            TEST_MODEL[i], np.mean(ap_values), np.mean(np.array(ap_values * weights)), np.mean(box_losses)
        ))

        plt.xlabel("recall", fontsize=14)
        plt.ylabel("precision", fontsize=14)
        plt.legend()

    plt.show()

    shutil.rmtree(TEMP_NAME)
    shutil.rmtree(RES_NAME)
