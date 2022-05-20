import os
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import torch
import torch.nn.functional
from tqdm import tqdm
from torchvision.utils import draw_bounding_boxes
from torch.utils.data import DataLoader

from data.classification_dataset import *
from my_utils.utils import show, make_dir, get_logger, load_checkpoint
from classifier_model import ClassifierModel
from my_utils.my_validation import *

# check if GPU is available
cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda:0" if cuda else "cpu")

if cuda:
    # define tensor type
    tensor = torch.cuda.FloatTensor
    # speedup fixed graph
    torch.backends.cudnn.benchmark = True
else:
    tensor = torch.FloatTensor


def train(
        output_dir: str,
        session_name: str,
        dataset_dir: str,
        val_dataset_dir: str,
        img_size: tuple,
        class_list: tuple,
        model_type: str,
        resize_mode: str,
        epochs: int,
        batch_size: int,
        shuffle: bool,
        num_workers: int,
        draw_img=False,
):
    # make dir for output dir
    output_dir = os.path.join(output_dir, session_name)
    make_dir(output_dir)

    # get the logger
    logger, log_path = get_logger(session_name, output_dir)

    train_dataset = MultiClassDataset(
        dataset_dir=dataset_dir,
        size=img_size,
        classes=class_list,
        resize_mode=resize_mode,
        use_random_augmentation=True,
    )
    persistent_workers = True if num_workers > 0 else False
    train_loader = DataLoader(
        dataset=train_dataset,
        shuffle=shuffle,
        num_workers=num_workers,
        batch_size=batch_size,
        # prefetch_factor=4,
        persistent_workers=persistent_workers,
    )
    val_dataset = MultiClassDataset(
        dataset_dir=val_dataset_dir,
        size=img_size,
        classes=class_list,
        resize_mode=resize_mode,
        use_random_augmentation=True,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        shuffle=shuffle,
        num_workers=num_workers,
        batch_size=batch_size,
        # prefetch_factor=4,
        persistent_workers=persistent_workers,
    )
    model = ClassifierModel(net_type=model_type, num_classes=len(class_list))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    loss = torch.nn.CrossEntropyLoss()
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    if cuda:
        model = torch.nn.DataParallel(model.cuda())
    else:
        model = torch.nn.DataParallel(model)

    # use checkpoints
    checkpoint_dir = os.path.join(output_dir, "saved_checkpoints")
    make_dir(checkpoint_dir)

    # get current epoch
    start_epoch = 0
    if Path(log_path).is_file():
        log = open(log_path, 'r')
        next_line = log.readline()
        while next_line:
            if "===EPOCH=FINISH===" in next_line:
                start_epoch += 1
            next_line = log.readline()

    if start_epoch != 0:
        # Load pretrained models
        logger.info('Loading previous checkpoint!')
        model, optimizer \
            = load_checkpoint(model, optimizer,
                              os.path.join(checkpoint_dir, "{}_param_{}.pkl".format(
                                  'model', start_epoch - 1)),
                              pickle_module=pickle, device=device, logger=logger)

    losses = []
    count = 0
    for epoch_idx in range(start_epoch, epochs):
        for imgs, labels in tqdm(train_loader, desc='epoch %d' % epoch_idx):
            # one_hot_labels = torch.zeros(batch_size, len(class_list)).scatter_(1, labels, 1)
            one_hot_labels = torch.nn.functional.one_hot(labels.type(torch.int64), len(class_list))
            if cuda:
                imgs = imgs.cuda()
                one_hot_labels = labels.cuda()
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            # print(targets)
            out = model(imgs)
            loss_ = loss(out, one_hot_labels)
            loss_.backward()
            optimizer.step()
            loss_value = loss_.detach()
            if cuda:
                loss_value = loss_value.cpu()
            loss_value = loss_value.item()
            losses.append(loss_value)
            count += 1
        loss_mean = sum(losses) / count
        logger.info("loss: %f" % loss_mean)

        with torch.no_grad():
            labels_list, prediction_list = [], []
            for imgs, labels in tqdm(val_loader, desc='epoch %d' % epoch_idx):
                labels = torch.nn.functional.one_hot(labels.type(torch.int64), len(class_list))
                model.eval()
                prediction = model(imgs)
                for i in range(labels.shape[0]):
                    labels_list.append(labels[i].tolist())
                    prediction_list.append(prediction[i].tolist())
            f1 = get_Weightedf1score(labels_list, prediction_list)
            logger.info("f1: %f" % f1)
            if draw_img:
                labels_tensor = torch.zeros(len(labels_list), *torch.tensor(labels_list[0]).shape)
                for i in range(len(labels_list)):
                    labels_tensor[i] = torch.tensor(labels_list[i])
                predictions_tensor = torch.zeros(len(prediction_list), *torch.tensor(prediction_list[0]).shape)
                for i in range(len(prediction_list)):
                    predictions_tensor[i] = torch.tensor(prediction_list[i])
                labels = class_list
                # validation_classifier(labels_tensor, predictions_tensor, labels)
                torch.save(labels_tensor, os.path.join(output_dir, 'labels.pt'))
                torch.save(predictions_tensor, os.path.join(output_dir, 'predictions.pt'))
                # print(predictions_tensor)
                # print(labels)
            # print(prediction)
            # print(loss_val)

        # save checkpoint
        gen_state_checkpoint = {
            'epoch': epoch_idx,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(gen_state_checkpoint,
                   os.path.join(checkpoint_dir, "{}_param_{}.pkl".format('model', epoch_idx)), pickle_module=pickle)

        logger.info("===EPOCH=FINISH===")


if __name__ == '__main__':
    train(
        output_dir=r'E:\my_files\programmes\python\detector_output',
        session_name='classifier-mask-mobileNet',
        dataset_dir=r'E:\my_files\programmes\python\dp_dataset\mask',
        val_dataset_dir=r'E:\my_files\programmes\python\dp_dataset\mask',
        img_size=(120, 120),
        class_list=MASK_CLASSES,
        model_type='mobileNet',
        resize_mode='stretch',
        epochs=10,
        batch_size=8,
        shuffle=True,
        num_workers=2,
        draw_img=False,
    )
    train(
        output_dir=r'E:\my_files\programmes\python\detector_output',
        session_name='classifier-sunglasses-mobileNet',
        dataset_dir=r'E:\my_files\programmes\python\dp_dataset\sunglasses',
        val_dataset_dir=r'E:\my_files\programmes\python\dp_dataset\sunglasses',
        img_size=(120, 120),
        class_list=SUNGLASSES_CLASSES,
        model_type='mobileNet',
        resize_mode='stretch',
        epochs=20,
        batch_size=8,
        shuffle=True,
        num_workers=2,
        draw_img=False,
    )
    '''
    output_dir: str,
    session_name: str,
    dataset_dir: str,
    img_size: tuple,
    class_list: tuple,
    model_type: str,
    resize_mode: str,
    epochs: int,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    '''
    '''
    train(
        output_dir=r'E:\my_files\programmes\python\detector_output',
        session_name='classifier-head-mobileNet-b8-',
        dataset_dir=r'E:\my_files\programmes\python\dp_dataset\head_classification\_train',
        val_dataset_dir=r'E:\my_files\programmes\python\dp_dataset\head_classification\_validation',
        img_size=(120, 120),
        class_list=HEAD_MOTIONS,
        model_type='mobileNet',
        resize_mode='stretch',
        epochs=200,
        batch_size=8,
        shuffle=True,
        num_workers=2,
        draw_img=True
    )
    train(
        output_dir=r'E:\my_files\programmes\python\detector_output',
        session_name='classifier-head-mobileNet-b4-',
        dataset_dir=r'E:\my_files\programmes\python\dp_dataset\head_classification\_train',
        val_dataset_dir=r'E:\my_files\programmes\python\dp_dataset\head_classification\_validation',
        img_size=(120, 120),
        class_list=HEAD_MOTIONS,
        model_type='mobileNet',
        resize_mode='stretch',
        epochs=200,
        batch_size=4,
        shuffle=True,
        num_workers=2,
        draw_img=True
    )
    train(
        output_dir=r'E:\my_files\programmes\python\detector_output',
        session_name='classifier-head-mobileNet-b2-',
        dataset_dir=r'E:\my_files\programmes\python\dp_dataset\head_classification\_train',
        val_dataset_dir=r'E:\my_files\programmes\python\dp_dataset\head_classification\_validation',
        img_size=(120, 120),
        class_list=HEAD_MOTIONS,
        model_type='mobileNet',
        resize_mode='stretch',
        epochs=200,
        batch_size=2,
        shuffle=True,
        num_workers=2,
        draw_img=True
    )
    '''
