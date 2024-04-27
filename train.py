import argparse
import os
import sys

import yaml
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from dataset.npy_datasets import NPYDatasets
from engine import *
from models.malunet import MALUNet
from transforms import train_transforms, test_transformers

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # "0, 1, 2, 3"

from utils import *

import warnings

warnings.filterwarnings("ignore")


def main(config):
    print('#----------Creating logger----------#')
    sys.path.append(config["work_dir"] + '/')
    checkpoint_dir = os.path.join(config["work_dir"], 'checkpoints')
    resume_model = os.path.join(checkpoint_dir, 'latest.pth')
    outputs = os.path.join(config["work_dir"], 'outputs')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(outputs):
        os.makedirs(outputs)

    print('#----------GPU init----------#')
    set_seed(config["seed"])
    gpu_ids = [0]  # [0, 1, 2, 3]
    torch.cuda.empty_cache()

    print('#----------Preparing dataset----------#')
    train_dataset = NPYDatasets(
        config,
        transformer=train_transforms(
            config["datasets"],
            input_size_h=config["input_size_h"],
            input_size_w=config["input_size_w"]
        ),
        train=True
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        pin_memory=True,
        num_workers=config["num_workers"]
    )
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

    model = torch.nn.DataParallel(model.cuda(), device_ids=gpu_ids, output_device=gpu_ids[0])

    print('#----------Prepareing loss, opt, sch and amp----------#')
    criterion = BceDiceLoss()
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)
    scaler = GradScaler()

    print('#----------Set other params----------#')
    min_loss = 999
    start_epoch = 1
    min_epoch = 1

    if os.path.exists(resume_model):
        print('#----------Resume Model and Other params----------#')
        checkpoint = torch.load(resume_model, map_location=torch.device('cpu'))
        model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        saved_epoch = checkpoint['epoch']
        start_epoch += saved_epoch
        min_loss, min_epoch, loss = checkpoint['min_loss'], checkpoint['min_epoch'], checkpoint['loss']

        log_info = f'resuming model from {resume_model}. resume_epoch: {saved_epoch}, min_loss: {min_loss:.4f}, min_epoch: {min_epoch}, loss: {loss:.4f}'
        print(log_info)

    print('#----------Training----------#')
    for epoch in range(start_epoch, config["epochs"] + 1):

        torch.cuda.empty_cache()

        train_one_epoch(
            train_loader,
            model,
            criterion,
            optimizer,
            scheduler,
            epoch,
            config,
            scaler=scaler
        )

        loss = val_one_epoch(
            val_loader,
            model,
            criterion,
            epoch,
            config
        )

        if loss < min_loss:
            torch.save(model.module.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))
            min_loss = loss
            min_epoch = epoch

        torch.save(
            {
                'epoch': epoch,
                'min_loss': min_loss,
                'min_epoch': min_epoch,
                'loss': loss,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, os.path.join(checkpoint_dir, 'latest.pth'))



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='./configs/config.yaml', help="config file path")
    parser.add_argument("--dataset_path", type=str, default=None, help="dataset path")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    with open(args.config, "r") as yaml_file:
        # 使用PyYAML加载YAML数据
        config = yaml.safe_load(yaml_file)
    config["dataset_path"] = args.dataset_path if args.dataset_path is not None else config["dataset_path"]

    main(config)
