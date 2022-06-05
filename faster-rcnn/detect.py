import torch
import torchvision
import argparse
import cv2
import numpy as np
import sys
import os
import time
import pynvml
import typing

sys.path.append('./')
import random
from util.save import get_exp_name, InputType, judge_file_type
from util.progress import process_bar

class Config:
    score : float       # threshold to display bbox
    class_num : int
    class_names : str
    device : str
    run_dir : str = "runs"
    save_txt : bool
    save_conf : bool
    nosave : bool

COLORS = (
    (255, 255, 255),     # reserve for background
    ( 89, 221,  64),
    (253, 111,  54),    
    (128,  30, 255),
    ( 92, 155, 255),
    ( 94, 108, 124),
    ( 28,  93, 232)
)

def random_color():
    b = random.randint(0, 255)
    g = random.randint(0, 255)
    r = random.randint(0, 255)
    return (b,g,r)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, help='image path/video path/image dir/video dir, 0 is webcam')
    parser.add_argument('--weights', default='default', help='weights, default to use torchvision')
    parser.add_argument('--device', default='cuda', help='device cpu or cuda')
    parser.add_argument('--score', type=float, default=0.8, help='objectness score threshold')

    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    
    parser.add_argument('--project', default="runs/detect", help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    args = parser.parse_args()
 
    return args

def detect_image(model, src_img : np.ndarray, names : dict, colormap : dict) -> np.ndarray:
    """inputs image BGR"""
    
    result = {
        "labels" : [],
        "bboxes" : [],
        "scores" : [],
        "inference_time" : 0
    }
 
    device = Config.device
    img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img / 255.).permute(2, 0, 1).float()
    if device == "cuda":
        img_tensor = img_tensor.cuda()
    s = time.time()
    with torch.no_grad():
        out = model([img_tensor])
    result['inference_time'] = time.time() - s

    boxes = out[0]['boxes']
    labels = out[0]['labels']
    scores = out[0]['scores']

    lw = max(round(sum(src_img.shape) / 2 * 0.003), 2)
    # tf = max(lw - 1, 1)
    tf = 2

    for idx in range(boxes.shape[0]):
        if scores[idx] >= Config.score:
            x1, y1, x2, y2 = boxes[idx][0], boxes[idx][1], boxes[idx][2], boxes[idx][3]
            p1 = (int(x1), int(y1))
            p2 = (int(x2), int(y2))

            yolo_x = (x1 + x2) / 2 / src_img.shape[1]
            yolo_y = (y1 + y2) / 2 / src_img.shape[0]
            yolo_w = (x2 - x1) / src_img.shape[1]
            yolo_h = (y2 - y1) / src_img.shape[0]

            result["bboxes"].append((float(yolo_x), float(yolo_y), float(yolo_w), float(yolo_h)))
            result["labels"].append(str(labels[idx].item()))
            result["scores"].append(scores[idx].item())

            name = names.get(str(labels[idx].item()))
            label = name + " " + str(round(scores[idx].item(), 2))
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

def handle_image(image_path : str, model : torch.nn.Module, names : dict, colormap : dict, save_dir : str) -> None:
    """handle image and save result to run/detect/expi/*"""
    src_img = cv2.imread(image_path)
    result = detect_image(model, src_img, names, colormap)
    print("[LOG] cost time {}ms".format(round(result["inference_time"] * 1000, 2)))
    image_name = os.path.basename(image_path)
    save_path = os.path.join(save_dir, image_name)
    if not Config.nosave:
        cv2.imwrite(save_path, result["image"])
        print("[LOG] save result to {}".format(save_path))
    if Config.save_txt:
        txt_name = image_name.split(".")[0] + ".txt"
        txt_path = os.path.join(save_dir, "labels")
        if not os.path.exists(txt_path):
            os.makedirs(txt_path)
        txt_path = os.path.join(txt_path, txt_name)
        fp = open(txt_path, "w", encoding="utf-8")
        for i in range(len(result["bboxes"])):
            fp.write("{} ".format(int(result["labels"][i]) - 1))
            box = result["bboxes"][i]
            fp.write("{} {} {} {}".format(box[0], box[1], box[2], box[3]))
            if Config.save_conf:
                fp.write(" {}".format(result['scores'][i]))
            fp.write("\n")
        fp.close()
        print("[LOG] save result to {}".format(txt_path))

def handle_video(video_path : str, model : torch.nn.Module, names : dict, colormap : dict, save_dir : str):
    assert os.path.exists(video_path)
    
    # TODO support more video format
    FORMAT = "mp4v"
    video_name = os.path.basename(video_path)
    save_path = os.path.join(save_dir, video_name)

    cap = cv2.VideoCapture(video_path)
    if not Config.nosave:
        fourcc = cv2.VideoWriter_fourcc(*FORMAT)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print("[LOG] current video information: FPS : {}, height : {}, width : {}".format(fps, height, width))
        out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
    
    cur = 1
    while (cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break 
        result = detect_image(model, frame, names, colormap)
        if not Config.nosave:
            out.write(result["image"])
        process_bar(cur, frames, prefix='[PROCRESS]')
        cur += 1
        
    cap.release()
    if not Config.nosave:
        out.release()
    cv2.destroyAllWindows()


def handle_webcam(model : torch.nn.Module, names : dict, colormap : dict):
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret is False:
            break
        result = detect_image(model, frame, names, colormap)
        cv2.imshow("webcam", result["image"])
        if cv2.waitKey(1) >= 0:
            break


def main():
    args = get_args()
    Config.device = args.device
    Config.score = args.score
    Config.nosave = args.nosave
    Config.save_txt = args.save_txt
    Config.save_conf = args.save_conf
        
    # Model creating
    print("[LOG] Initialise model")
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    if args.weights == "default":
        names = {'0': 'background', '1': 'person', '2': 'bicycle', '3': 'car', '4': 'motorcycle', '5': 'airplane', '6': 'bus', '7': 'train', '8': 'truck', '9': 'boat', '10': 'traffic light', '11': 'fire hydrant', '13': 'stop sign', '14': 'parking meter', '15': 'bench', '16': 'bird', '17': 'cat', '18': 'dog', '19': 'horse', '20': 'sheep', '21': 'cow', '22': 'elephant', '23': 'bear', '24': 'zebra', '25': 'giraffe', '27': 'backpack', '28': 'umbrella', '31': 'handbag', '32': 'tie', '33': 'suitcase', '34': 'frisbee', '35': 'skis', '36': 'snowboard', '37': 'sports ball', '38': 'kite', '39': 'baseball bat', '40': 'baseball glove', '41': 'skateboard', '42': 'surfboard', '43': 'tennis racket', '44': 'bottle', '46': 'wine glass', '47': 'cup', '48': 'fork', '49': 'knife', '50': 'spoon', '51': 'bowl', '52': 'banana', '53': 'apple', '54': 'sandwich', '55': 'orange', '56': 'broccoli', '57': 'carrot', '58': 'hot dog', '59': 'pizza', '60': 'donut', '61': 'cake', '62': 'chair', '63': 'couch', '64': 'potted plant', '65': 'bed', '67': 'dining table', '70': 'toilet', '72': 'tv', '73': 'laptop', '74': 'mouse', '75': 'remote', '76': 'keyboard', '77': 'cell phone', '78': 'microwave', '79': 'oven', '80': 'toaster', '81': 'sink', '82': 'refrigerator', '84': 'book', '85': 'clock', '86': 'vase', '87': 'scissors', '88': 'teddybear', '89': 'hair drier', '90': 'toothbrush'}
    else:
        assert os.path.exists(args.weights)
        print("[LOG] find {}, attempt loading".format(args.weights))
        model_dict = torch.load(args.weights)
        class_num = model_dict["class_num"]
        names = {str(i) : name for i, name in enumerate(model_dict["class_name"])}  
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
            in_channels=in_features, num_classes=class_num)      # class_n + background
        model.load_state_dict(model_dict["state_dict"])
        print("[LOG] finish loading, class_name : {}".format(model_dict["class_name"]))

        Config.class_names = model_dict["class_name"]
        Config.class_num = class_num

    if len(COLORS) < len(names):
        colormap = { names[k] : random_color() for k in names }
    else:
        colormap = { names[k] : c for k, c in zip(names, COLORS) }

    if Config.device == "cuda":
        model = model.cuda()
    model.eval()

    print("[LOG] everything is ready, device : {}".format(Config.device))

    # determine save dir
    detect_root = args.project
    if args.name == "exp":
        exp_name = get_exp_name(os.listdir(detect_root))
        save_dir = os.path.join(detect_root, exp_name)
    else:
        save_dir = os.path.join(detect_root, args.name)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    

    input_type = judge_file_type(args.source)
    if input_type == InputType.WEBCAM:
        handle_webcam(model, names, colormap)
    elif input_type == InputType.IMAGE:
        handle_image(args.source, model, names, colormap, save_dir)
    elif input_type == InputType.VIDEO:
        handle_video(args.source, model, names, colormap, save_dir)
    elif input_type == InputType.DIR:
        for file in os.listdir(args.source):
            file_path = os.path.join(args.source, file)
            input_type = judge_file_type(file_path)
            if input_type == InputType.IMAGE:
                handle_image(file_path, model, names, colormap, save_dir)
            elif input_type == InputType.VIDEO:
                handle_video(file_path, model, names, colormap, save_dir)
    
if __name__ == "__main__":
    main()