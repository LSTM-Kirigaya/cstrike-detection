## 训练YOLO v5

```bash
$cd yolov5
$python train.py --img 640 --batch 4 --epochs 25 --data CS.yaml --weights .\models\pretrained\yolov5m.pt
```

其中`.\models\pretrained\yolov5m.pt`为下载的预训练模型，请自行前往YOLOv5官方仓库下载。

## YOLO v5推理

```bash
$python .\detect.py --weights .\runs\train\exp\weights\best.pt --img 640 --device 0 --source <图片\视频\图片文件夹>
```

`.\runs\train\exp\weights\best.pt`是训练得到的模型，训练得到的模型都存储在exp文件夹下，里面还同时存储着标签分布等可视化结果