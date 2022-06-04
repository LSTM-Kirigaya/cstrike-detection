import torch
import torchvision
from torch.utils.data import DataLoader

import argparse
from easydict import EasyDict
import yaml
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from util.read_data import read_from_meta_dataset
from util.data_loader import CS
from util.save import get_exp_name, find_best_model

class Config:
    train : str
    val : str
    class_num : int
    class_names : str
    epochs : int
    batch : int
    device : str
    run_dir : str = "runs"

def get_parser() -> EasyDict:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help="yaml path")
    parser.add_argument('--weights', default="default", help="weights model")
    parser.add_argument('--device', default="cuda")
    parser.add_argument('--batch', default=4, type=int, help="batch size")
    parser.add_argument('--epochs', default=10, type=int, help="times to train the whole model")
    return EasyDict(vars(parser.parse_args()))

def get_model(args : EasyDict) -> torch.nn.Module:
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_channels=in_features, num_classes=Config.class_num)      # class_n + background

    if args.weights != "default":
        assert os.path.exists(args.weights)
        model.load_state_dict(torch.load(f=args.weights))
    if args.device == "cuda":
        model = model.cuda()
    return model


def main():
    args = get_parser()
    assert os.path.exists(args.data)
    assert isinstance(args.data, str)
    assert args.data.endswith(".yaml") or args.data.endswith(".yml")
    with open(args.data, "r", encoding="utf-8") as fp:
        data_config : EasyDict = EasyDict(yaml.load(fp, yaml.Loader))
    
    Config.train = os.path.join(data_config.path, data_config.train)
    Config.val = os.path.join(data_config.path, data_config.val)
    Config.class_num = data_config.nc + 1
    Config.class_names = ["background"] + data_config.names
    Config.epochs = args.epochs
    Config.batch = args.batch
    Config.device = args.device

    assert os.path.exists(Config.train)
    assert os.path.exists(Config.val)

    # dir
    target_save_dir = os.path.join(Config.run_dir, "train")
    exp_name = get_exp_name(os.listdir(target_save_dir))
    target_save_dir = os.path.join(target_save_dir, exp_name)
    weights_dir = os.path.join(target_save_dir, "weights")
    os.makedirs(target_save_dir)
    os.makedirs(weights_dir)
    
    # acquire model
    model = get_model(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # acquire dataset
    train, test = read_from_meta_dataset(Config.train, Config.val)
    image_root_dir = os.path.join(data_config.path, "images")
    train_dataset = CS(train, image_root_dir)
    test_dataset = CS(test, image_root_dir)

    # construct dataloader
    train_data_loader = DataLoader(
        dataset=train_dataset,
        batch_size=Config.batch,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda batch : tuple(zip(*batch))
    )

    test_data_loader = DataLoader(
        dataset=test_dataset,
        batch_size=Config.batch,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda batch : tuple(zip(*batch))
    )

    train_losses = []
    val_losses = []

    for epoch in range(Config.epochs):
        model.train()
        epoch_train_losses = []
        progress_bar = tqdm(train_data_loader, desc="start train")
        for images, targets, _ in progress_bar:
            images = list(image.to(Config.device) for image in images)
            targets = [{k: v.to(Config.device) for k, v in t.items()} for t in targets]
            
            loss_dict : dict = model(images, targets)
            losses : torch.Tensor = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            progress_bar.set_description_str(desc="EPOCH : {}, batch size {},cur val loss {}".format(
                    str(epoch) + "/" + str(Config.epochs - 1), Config.batch, losses.item()))
            epoch_train_losses.append(losses.item())
            
        epoch_val_losses = []
        progress_bar = tqdm(test_data_loader, desc="start val")
        with torch.no_grad():
            for images, targets, _ in progress_bar:
                images = list(image.to(Config.device) for image in images)
                targets = [{k: v.to(Config.device) for k, v in t.items()} for t in targets]
                
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                progress_bar.set_description_str(desc="EPOCH : {}, batch size {},cur val loss {}".format(
                    str(epoch) + "/" + str(Config.epochs - 1), Config.batch, losses.item()))
                epoch_val_losses.append(losses.item())
        
        train_losses.append(np.mean(epoch_train_losses))
        val_losses.append(np.mean(epoch_val_losses))
        print("[LOG] train loss : {} test loss : {}".format(train_losses[-1], val_losses[-1]))
        torch.save({
            "state_dict" : model.state_dict(),
            "class_num" : Config.class_num,
            "class_name" : Config.class_names
        }, os.path.join(weights_dir, "{}_{}".format(epoch, round(val_losses[-1], 5))))


    # find best model to save and other's removal
    find_best_model(weights_dir)


    plt.figure(dpi=120)
    plt.xlabel("epoch")
    plt.ylabel("loss_val")
    plt.plot(train_losses, '-o', label="train")
    plt.plot(val_losses, '-o', label="val")
    plt.legend()
    plt.savefig(os.path.join(target_save_dir, "loss_curve.png"))

    torch.save({
        "state_dict" : model.state_dict(),
        "class_num" : Config.class_num,
        "class_name" : Config.class_names
    }, os.path.join(weights_dir, "last.pth"))

if __name__ == "__main__":
    main()