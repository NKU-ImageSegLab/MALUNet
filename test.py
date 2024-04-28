import os

import torch
from torch.utils.data import DataLoader

from dataset.npy_datasets import NPYDatasets
from engine import test_one_epoch
from models.malunet import MALUNet
from transforms import test_transformers
from utils import set_seed


def main(config):
    checkpoint_dir = os.path.join(config["work_dir"], 'checkpoints')
    outputs = os.path.join(config["work_dir"], 'outputs')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(outputs):
        os.makedirs(outputs)

    print('#----------GPU init----------#')
    set_seed(config["seed"])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.cuda.empty_cache()

    print('#----------Preparing dataset----------#')
    val_dataset = NPYDatasets(
        config,
        transformer=test_transformers(
            config["datasets"],
            input_size_h=config["input_size_h"],
            input_size_w=config["input_size_w"]
        ),
        train=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=config["num_workers"],
        drop_last=True
    )

    print('#----------Prepareing Models----------#')
    model_cfg = config["model_config"]
    model = MALUNet(
        num_classes=model_cfg['num_classes'],
        input_channels=model_cfg['input_channels'],
        c_list=model_cfg['c_list'],
        split_att=model_cfg['split_att'],
        bridge=model_cfg['bridge']
    )

    model = model.to(device)

    if os.path.exists(os.path.join(checkpoint_dir, 'best.pth')):
        print('#----------Testing----------#')
        best_weight = torch.load(config["work_dir"] + 'checkpoints/best.pth', map_location=torch.device('cpu'))
        model.load_state_dict(best_weight)
        test_one_epoch(
            val_loader,
            model,
            config,
        )