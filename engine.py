import os

import numpy as np
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast
from sklearn.metrics import confusion_matrix

from metrics import get_binary_metrics, MetricsResult
from utils import save_imgs
import imageio

def train_one_epoch(
        train_loader,
        model,
        criterion,
        optimizer,
        scheduler,
        epoch,
        config,
        scaler=None
):
    '''
    train model for one epoch
    '''
    # switch to train mode
    model.train()

    loss_list = []
    metrics = get_binary_metrics()
    for iter, data in tqdm(
            iterable=enumerate(train_loader),
            desc=f"{config['datasets']} Training [{epoch}/{config['epochs']}]",
            total=len(train_loader)
    ):
        optimizer.zero_grad()
        images, targets, _ = data
        images, targets = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).float()
        if config["amp"]:
            with autocast():
                out = model(images)
                loss = criterion(out, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(images)
            loss = criterion(out, targets)
            loss.backward()
            optimizer.step()
        metrics.update(out, targets.int())
        loss_list.append(loss.item())

        now_lr = optimizer.state_dict()['param_groups'][0]['lr']
    result = MetricsResult(metrics.compute())
    print(result.to_log('Train', epoch - 1, config["epochs"] + 1, np.mean(loss_list)))
    scheduler.step()


def val_one_epoch(test_loader,
                  model,
                  criterion,
                  epoch,
                  config):
    # switch to evaluate mode
    model.eval()
    preds = []
    gts = []
    loss_list = []
    with torch.no_grad():
        metrics = get_binary_metrics()
        for data in tqdm(
                iterable=test_loader,
                desc=f"{config['datasets']} Val [{epoch}/{config['epochs']}]",
                total=len(test_loader)
        ):
            img, msk, _ = data
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()
            out = model(img)
            metrics.update(out, msk.int())
            loss = criterion(out, msk)
            loss_list.append(loss.item())
            gts.append(msk.squeeze(1).cpu().detach().numpy())
            if type(out) is tuple:
                out = out[0]
            out = out.squeeze(1).cpu().detach().numpy()
            preds.append(out)
        result = MetricsResult(metrics.compute())
        print(result.to_log('Val', epoch - 1, config["epochs"] + 1, np.mean(loss_list)))

    return np.mean(loss_list)


def test_one_epoch(
        test_loader,
        model,
        output_folder_path,
):
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        metrics = get_binary_metrics()
        for i, data in enumerate(tqdm(test_loader)):
            img, msk, image_names = data
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()
            out = model(img)
            metrics.update(out, msk.int())

            msk_pred_output = out.cpu().detach().numpy()[0, 0] > 0.5
            msk_pred_output = (msk_pred_output * 255).astype(np.uint8)
            # name = 'test_'+str(index)+'.png'
            # print(name)
            total_length = len(image_names)
            for i in range(total_length):
                name = image_names[i] + '.png'
                if total_length == 1:
                    predict_image = msk_pred_output
                else:
                    predict_image = msk_pred_output[i]
                imageio.imwrite(os.path.join(output_folder_path, name), predict_image)
