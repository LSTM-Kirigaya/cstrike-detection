import os
import typing

def parse_yolo_label_file(label_path : str) -> typing.List[typing.Tuple[float]]:
    targets = []
    if not os.path.exists(label_path):      # this is a negative label
        return targets

    for label_line in open(label_path, "r", encoding="utf-8"):
        label, x, y, w, h = label_line.split()
        label = int(label) + 1      # start from 1, 0 is background
        x = float(x)
        y = float(y)
        w = float(w)
        h = float(h)
        x_min = x - w / 2
        x_max = x + w / 2
        y_min = y - h / 2
        y_max = y + h / 2
        targets.append((label, x_min, y_min, x_max, y_max))
    return targets

def read_from_meta_dataset(traintxt : str, testtxt : str) -> tuple:
    assert os.path.exists(traintxt)
    assert os.path.exists(testtxt)

    train_meta = []
    test_meta = []
    for line in open(traintxt, "r", encoding="utf-8"):
        path = line.strip()
        file = os.path.basename(path)
        image_id = file.rstrip(".png")
        label_path = path.replace("png", "txt").replace("images", "labels")
        labels = parse_yolo_label_file(label_path)
        train_meta.append((image_id, labels))
    
    for line in open(testtxt, "r", encoding="utf-8"):
        path = line.strip()
        file = os.path.basename(path)
        image_id = file.rstrip(".png")
        label_path = path.replace("png", "txt").replace("images", "labels")
        labels = parse_yolo_label_file(label_path)
        test_meta.append((image_id, labels))
    
    return train_meta, test_meta