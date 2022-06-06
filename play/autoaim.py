import argparse
import os
import cv2
import numpy as np
import torch
import torchvision
import win32api
import win32con
from screen_shot import shot_display, init_shot, move_here
from cv_util import resize 

class Config:
    device : str = "cpu"
    score : float = 0.6
    class_names : list
    class_num : int
    debug : bool
    auto_aim : bool
    auto_fire : bool

COLORS = (
    (255, 255, 255),     # reserve for background
    ( 89, 221,  64),
    (253, 111,  54),    
    (128,  30, 255),
    ( 92, 155, 255),
    ( 94, 108, 124),
    ( 28,  93, 232)
)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights')
    parser.add_argument('--device', default="cpu")
    parser.add_argument('--score', default=0.6, type=float)
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--auto-aim', action="store_true")
    parser.add_argument('--auto-fire', action="store_true")

    return parser.parse_args()


def detect_image(model, src_img : np.ndarray, names : dict, colormap : dict) -> np.ndarray:
    """inputs image BGR"""

    device = Config.device
    img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img / 255.).permute(2, 0, 1).float()
    if device == "cuda":
        img_tensor = img_tensor.cuda()
    with torch.no_grad():
        out = model([img_tensor])

    result = {
        "image" : None,
        "target_bbox" : None,
        "target_area" : None
    }
    boxes = out[0]['boxes']
    labels = out[0]['labels']
    scores = out[0]['scores']

    lw = max(int(round(sum(src_img.shape) / 2 * 0.003)), 2)
    # tf = max(lw - 1, 1)
    tf = 2

    for idx in range(boxes.shape[0]):
        if scores[idx] >= Config.score:
            x1, y1, x2, y2 = boxes[idx][0], boxes[idx][1], boxes[idx][2], boxes[idx][3]

            name = names.get(str(labels[idx].item()))
            area = (y2 - y1) * (x2 - x1)
            # if name == 'T' or name == 'C':
            if name == 'T':
                if result["target_area"] is None or result["target_area"] < area:
                    result["target_area"] = area
                    result["target_bbox"] = (int(x1), int(y1), int(x2), int(y2))
            
            
            if Config.debug:
                label = name + " " + str(round(scores[idx].item(), 2))
                p1 = (int(x1), int(y1))
                p2 = (int(x2), int(y2))
                w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
                cv2.rectangle(src_img, p1, p2, colormap[name], thickness=lw, lineType=cv2.LINE_AA)
                outside = p1[1] - h >= 3
                p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
                cv2.rectangle(src_img, p1, p2, colormap[name], -1, cv2.LINE_AA)  # filled
                cv2.putText(
                    src_img,
                    label, 
                    (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                    0,
                    fontScale=lw / 3,
                    color=(255, 255, 255),
                    thickness=tf,
                    lineType=cv2.LINE_AA
                )
    result["image"] = src_img

    return result


if __name__ == "__main__":
    args = get_args()
    assert os.path.exists(args.weights)
    Config.device = args.device
    Config.score = args.score
    Config.debug = args.debug
    Config.auto_aim = args.auto_aim
    Config.auto_fire = args.auto_fire

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model_dict = torch.load(args.weights)
    class_num = model_dict["class_num"]
    names = {str(i) : name for i, name in enumerate(model_dict["class_name"])}  
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_channels=in_features, num_classes=class_num)      # class_n + background
    model.load_state_dict(model_dict["state_dict"])
    if Config.device == "cuda":
        model = model.cuda()
    model.eval()
    print("[LOG] finish loading, class_name : {}".format(model_dict["class_name"]))

    Config.class_names = model_dict["class_name"]
    Config.class_num = class_num
    colormap = { names[k] : c for k, c in zip(names, COLORS) }

    pos = init_shot()

    while True:
        image = shot_display()
        result = detect_image(model, image, names, colormap)
        if Config.debug:
            cv2.imshow("result", resize(result["image"], width=600))
            if cv2.waitKey(1) >= 0:
                exit(-1)
        
        alpha = 0.8
        
        if result["target_bbox"]:
            x1, y1, x2, y2 = result["target_bbox"]
            x1 += pos[0]
            y1 += pos[1]
            x2 += pos[0]
            y2 += pos[1]
            print(x1, y1, x2, y2, pos)
            cx = int((x1 + x2) / 2)
            cy = int(y1 * alpha + y2 * (1 - alpha))
            if Config.auto_aim:
                move_here((cx, cy))
                print(cx, cy)
                if Config.auto_fire:
                    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0)
                    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0)
        else:
            # win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0)
            pass