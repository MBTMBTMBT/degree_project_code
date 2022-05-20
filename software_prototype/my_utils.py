import logging
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision.transforms import functional as F


def load_checkpoint(model, optimizer, filename, pickle_module, device, logger):
    if os.path.isfile(filename):
        logger.info("Loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(
            filename, pickle_module=pickle_module, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("Loaded checkpoint '{}' (epoch {})"
                    .format(filename, checkpoint['epoch']))
    else:
        logger.critical('Checkpoint {} does not exist.'.format(filename))
    return model, optimizer


def load_model(model, filename, pickle_module, device):
    if os.path.isfile(filename):
        print("Loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(
            filename, pickle_module=pickle_module, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        print("Loaded checkpoint '{}' (epoch {})".format(filename, checkpoint['epoch']))
    else:
        print('Checkpoint {} does not exist.'.format(filename))
    return model


def load_checkpoint_cancel_parallel(model, optimizer, filename, pickle_module, device, logger):
    if os.path.isfile(filename):
        logger.info("Loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(
            filename, pickle_module=pickle_module, map_location=device)
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        # load params
        model.load_state_dict(new_state_dict)
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("Loaded checkpoint '{}' (epoch {})"
                    .format(filename, checkpoint['epoch']))
    else:
        logger.critical('Checkpoint {} does not exist.'.format(filename))
    return model, optimizer


def load_model_cancel_parallel(model, filename, pickle_module, device):
    if os.path.isfile(filename):
        print("Loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(
            filename, pickle_module=pickle_module, map_location=device)
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        # load params
        model.load_state_dict(new_state_dict)
        print("Loaded checkpoint '{}' (epoch {})".format(filename, checkpoint['epoch']))
    else:
        print('Checkpoint {} does not exist.'.format(filename))
    return model


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


def get_logger(session_name: str, log_dir: str):
    logger = logging.getLogger(name=session_name)
    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler(
        sys.stdout)  # stderr output to console
    log_path = os.path.join(log_dir, session_name + ".txt")
    file_handler = logging.FileHandler(log_path, mode='a')
    formatter = logging.Formatter(
        '%(asctime)s: %(message)s', "%y%b%d-%H:%M:%S")
    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger, log_path


def make_dir(absolute_dir: str):
    if len(absolute_dir.split('.')) > 1:  # if this is a file, use its parent; don't use file without suffix
        absolute_dir = os.path.abspath(os.path.dirname(absolute_dir) + os.path.sep + ".")
    if Path(absolute_dir).is_dir():
        return
    parent_dir = os.path.abspath(os.path.dirname(absolute_dir) + os.path.sep + ".")
    if Path(parent_dir).is_dir():
        os.mkdir(absolute_dir)
    else:
        make_dir(parent_dir)


def read_image(img_path) -> torch.Tensor:
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = np.transpose(np.array(img), (2, 0, 1))
    img_tensor = torch.tensor(img_tensor).type(torch.float32) / 255
    return img_tensor
