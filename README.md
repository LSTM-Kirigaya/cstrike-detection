# Cstrike Detection
![](https://img.shields.io/badge/Framework-PyTorch-brightgreen) 
![](https://img.shields.io/badge/Python_Version-3.7-blue) 
![](https://img.shields.io/badge/Task-Object_Detection-purple)
![](https://img.shields.io/badge/Scene-CS1.6-yellow)

## Motivation
The project is aimed at constructing automatic aiming system based on object detection algorithm, in order to have fun in `CS 1.6`.

## TODO

- [x] Finish training and test of `YOLO v5` and `Faster-RCNN`.
- [ ] Finish deployment pipeline.
- [ ] Construct automatic aiming system and test.

## Prepare for dataset
### Step 1. download dataset
I have published my home made dataset in [Kaggle](https://www.kaggle.com/datasets/lstmkirigaya/cstrike-detection). If you want to launch my project, please download the dataset first and unzip to folder `dataset` in root path.

```
root
|--datasets
|--faster-rcnn
|--yolov5
...
```

### step 2. make train data and test data
run python script in console:
```bash
$python script/make_meta.py -t 20
```
`-t 20` means size of test dataset is 20. This is the default value.
 
# Usage of detection model

Traning config of two algorithm is almost the same.

## YOLO v5
### train
```bash
$cd yolov5
$python train.py --img 640 --batch 4 --epochs 25 --data "CS.yaml" --weights "models/pretrained/yolov5m.pt"
```

Where `models/pretrained/yolov5m.pt` is the pretrained weight and will determine the specific structure of YOLO v5(`yolov5m` means the middle version of YOLO v5)

### detect

```bash
$python detect.py --weights "runs/train/exp/weights/best.pt" --img 640 --device 0 --source <media path>
```

Where `runs/train/exp/weights/best.pt` is the finetone weight and `<media path>` can be path of image, video or the folder of both. If `<media path>` is set to 0, then webcam will be used.


## Faster-RCNN
### train

```bash
python train.py --weights "default" --data "CS.yaml" --epochs 20 --batch 4
```

`weights` is set to "default" in default, which means use pretrained faster-rcnn backbone. If you want to use custom weights, please set the argument. Other arguments are the same as YOLO's `train.py`.

### detect

```bash
$python detect.py --weights "runs/train/exp/weights/best.pt" --score 0.6 --device cuda --source <media path>
```

Where `runs/train/exp/weights/best.pt` is the finetone weight and `<media path>` can be path of image, video or the folder of both. If `<media path>` is set to 0, then webcam will be used.