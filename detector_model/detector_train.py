import os
from pathlib import Path
import pickle
from tkinter.messagebox import NO
from torchvision.models.detection import \
    fasterrcnn_mobilenet_v3_large_320_fpn, \
    fasterrcnn_resnet50_fpn, \
    fasterrcnn_mobilenet_v3_large_fpn, \
    retinanet_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
# from torchvision.models.detection.retinanet import
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from torchvision.utils import draw_bounding_boxes
from torch.utils.data import DataLoader

from data.voc_dataset import VOCDataset, CLASS_NAMES
from my_utils.utils import show, make_dir, get_logger, load_checkpoint
from tensorflow import summary
import tensorflow as tf

# check if GPU is available
cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda:0" if cuda else "cpu")
gpus = tf.config.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

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
        img_size: tuple,
        resize_mode: str,
        epochs: int,
        batch_size: int,
        shuffle: bool,
        num_workers: int,
        model_type: str,
        learning_rate: float,
        trainable_backbone_layers=None,  # None or int
        save_freq=5,
):
    # make dir for output dir
    output_dir = os.path.join(output_dir, session_name)
    make_dir(output_dir)

    # get the logger
    logger, log_path = get_logger(session_name, output_dir)

    # create tensorboard loggers
    tensorboard_dir = os.path.join(output_dir, 'tensorboard')
    tb_train_dir = os.path.join(tensorboard_dir, "train")
    tb_val_dir = os.path.join(tensorboard_dir, "val")
    make_dir(tb_train_dir)
    make_dir(tb_val_dir)
    train_summary_writer = summary.create_file_writer(tb_train_dir)
    val_summary_writer = summary.create_file_writer(tb_val_dir)

    train_dataset = VOCDataset(
        dataset_dir=dataset_dir,
        size=img_size,
        resize_mode=resize_mode,
        use_random_augmentation=True,
    )

    persistent_workers = True if num_workers > 0 else False
    train_loader = DataLoader(
        dataset=train_dataset,
        shuffle=shuffle,
        num_workers=num_workers,
        batch_size=1,
        # prefetch_factor=4,
        persistent_workers=persistent_workers,
    )
    val_dataset = VOCDataset(
        dataset_dir=dataset_dir,  # todo: change this!!!
        size=img_size,
        resize_mode=resize_mode,
        use_random_augmentation=False,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        shuffle=False,
        num_workers=num_workers * 2,
        batch_size=1,
        # prefetch_factor=4,
        persistent_workers=persistent_workers,
    )

    model = None
    if model_type == 'mobilenet':
        model = fasterrcnn_mobilenet_v3_large_fpn(
            pretrained=False,
            progress=True,
            num_classes=train_dataset.num_classes,
            pretrained_backbone=True,
            trainable_backbone_layers=trainable_backbone_layers,
        )
    elif model_type == 'mobilenet320':
        model = fasterrcnn_mobilenet_v3_large_320_fpn(
            pretrained=False,
            progress=True,
            num_classes=train_dataset.num_classes,
            pretrained_backbone=True,
            trainable_backbone_layers=trainable_backbone_layers,
        )
    elif model_type == 'resnet':
        '''
        model = fasterrcnn_resnet50_fpn(
            pretrained=False,
            progress=True,
            num_classes=train_dataset.num_classes,
            pretrained_backbone=True,
            trainable_backbone_layers=trainable_backbone_layers,
        )
        '''
        # load Faster RCNN pre-trained model
        model = fasterrcnn_resnet50_fpn(pretrained=True)
        # get the number of input features
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # define a new head for the detector with required number of classes
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, train_dataset.num_classes)
    elif model_type == 'retina':
        model = retinanet_resnet50_fpn(
            pretrained=False,
            progress=True,
            num_classes=train_dataset.num_classes,
            pretrained_backbone=True,
            trainable_backbone_layers=trainable_backbone_layers,
        )
    # params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0005)
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0005)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    if cuda:
        # model = torch.nn.DataParallel(model.cuda())
        model = model.cuda()
    # else:
    #     model = torch.nn.DataParallel(model)

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

    start_epoch = start_epoch * save_freq

    if start_epoch != 0:
        # Load pretrained models
        logger.info('Loading previous checkpoint!')
        model, optimizer \
            = load_checkpoint(model, optimizer,
                              os.path.join(checkpoint_dir, "{}_param_{}.pkl".format(
                                  'model', start_epoch - 1)),
                              pickle_module=pickle, device=device, logger=logger)

    loss_dict = None
    prediction = None

    for epoch_idx in range(start_epoch, epochs):
        count = 0
        images, targets = [], []
        loss_sum_dict = {}
        loss_count = 0
        skip = 0
        for img, bbox_ref, labels in tqdm(train_loader, desc='epoch %d' % epoch_idx):
            bbox_ref = torch.squeeze(bbox_ref, dim=0)
            img = torch.squeeze(img, dim=0)
            labels = torch.squeeze(labels, dim=0)
            if cuda:
                bbox_ref = bbox_ref.cuda()
                img = img.cuda()
                labels = labels.cuda()
            # if labels.ndim == 0 and labels.item() == 0:
            #     print('continue')
            #     continue
            if labels[0] == 0:
                skip += 1
                continue
            count += 1
            target = {
                'boxes': bbox_ref,
                'labels': labels,
            }
            images.append(img)
            targets.append(target)
            if count % batch_size == 0 or len(train_dataset) - count <= batch_size:
                # total_losses = None
                # for i in range(batch_size):
                try:
                    model.train()
                    model.zero_grad()
                    optimizer.zero_grad()
                    # print(targets)
                    loss_dict = model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
                    for each_key in loss_dict.keys():
                        if each_key not in loss_sum_dict.keys():
                            loss_sum_dict[each_key] = loss_dict[each_key].detach().cpu().item()
                        else:
                            loss_sum_dict[each_key] += loss_dict[each_key].detach().cpu().item()
                        loss_count += 1
                    # if total_losses is None:
                    #     total_losses = losses
                    # else:
                    #     total_losses += losses
                    # total_losses /= batch_size
                    # total_losses.backward()
                    losses.backward()
                    optimizer.step()
                except Exception as e:
                    print(e)
                # print(output)
                n = len(train_dataset)
                if count < len(train_dataset) - 1:
                    images, targets = [], []
        print('skipped: %d' % skip)
        if cuda:
            total_losses, losses = None, None
            bbox_ref = None
            img = None
            labels = None
            images, targets = None, None
            torch.cuda.empty_cache()

        # print(loss_dict)
        loss_mean_dict = loss_sum_dict.copy()
        for each_key in loss_mean_dict.keys():
            loss_mean_dict[each_key] /= loss_count
        logger.info('===Mean=losses===')
        logger.info(str(loss_mean_dict))
        with train_summary_writer.as_default():
            for each_key in loss_mean_dict.keys():
                summary.scalar(each_key, loss_mean_dict[each_key], step=epoch_idx)
        loss_mean_dict = None
        loss_sum_dict = None

        # validation
        val_sum_dict = {}
        val_count = 0
        count = 0
        skip = 0
        images = []
        targets = []
        with torch.no_grad():
            for img, bbox_ref, labels in tqdm(val_loader, desc='epoch %d' % epoch_idx):
                bbox_ref = torch.squeeze(bbox_ref, dim=0)
                img = torch.squeeze(img, dim=0)
                labels = torch.squeeze(labels, dim=0)
                if cuda:
                    bbox_ref = bbox_ref.cuda()
                    img = img.cuda()
                    labels = labels.cuda()
                if labels[0] == 0:
                    skip += 1
                    continue
                count += 1
                target = {
                    'boxes': bbox_ref,
                    'labels': labels,
                }
                images.append(img)
                targets.append(target)
                # try:
                # model.eval()
                # model.zero_grad()
                loss_dict = model(images, targets)
                for each_key in loss_dict.keys():
                    if each_key not in val_sum_dict.keys():
                        val_sum_dict[each_key] = loss_dict[each_key].detach().cpu().item()
                    else:
                        val_sum_dict[each_key] += loss_dict[each_key].detach().cpu().item()
                    val_count += 1
                # print(prediction)
                # except Exception as e:
                #     print(e)
                images = []
                targets = []
            print('skipped: %d' % skip)
            if cuda:
                targets = None
                images = None
                torch.cuda.empty_cache()

        # print(loss_dict)
        val_mean_dict = val_sum_dict.copy()
        for each_key in val_mean_dict.keys():
            val_mean_dict[each_key] /= loss_count
        logger.info('===Mean=losses=val===')
        logger.info(str(val_mean_dict))
        with val_summary_writer.as_default():
            for each_key in val_mean_dict.keys():
                summary.scalar(each_key, val_mean_dict[each_key], step=epoch_idx)

        # save checkpoint
        gen_state_checkpoint = {
            'epoch': epoch_idx,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        if (epoch_idx + 1) % save_freq == 0:
            torch.save(gen_state_checkpoint,
                       os.path.join(checkpoint_dir, "{}_param_{}.pkl".format('model', epoch_idx)), pickle_module=pickle)
            logger.info("===EPOCH=FINISH===")

    # predicted
    '''
    assert prediction is not None
    bbox_ref = prediction[0]['boxes']
    labels = prediction[0]['labels']
    scores = prediction[0]['scores']
    bbox_ones = torch.ones_like(bbox_ref)
    bbox_ref = torch.where(bbox_ref > 1, bbox_ones, bbox_ref)
    bbox = bbox_ref.detach().clone()
    if len(bbox.shape) == 1:
        bbox = bbox.unsqueeze(dim=0)
        bbox_ref = bbox_ref.unsqueeze(dim=0)
    bbox[:, 0] = bbox_ref[:, 0] * 399
    bbox[:, 1] = bbox_ref[:, 1] * 299
    bbox[:, 2] = bbox_ref[:, 2] * 399
    bbox[:, 3] = bbox_ref[:, 3] * 299
    labels = torch.squeeze(labels).tolist()
    labels = [CLASS_NAMES[l] for l in labels] if isinstance(labels, list) else [CLASS_NAMES[labels]]
    # bbox = torch.unsqueeze(bbox[0], dim=0)
    # labels = [labels[0]]
    scores = torch.squeeze(scores).tolist()
    scores = scores if isinstance(scores, list) else [scores]
    labels = [labels[l] + " %.2f" % scores[l] for l in range(len(labels))]
    result = draw_bounding_boxes((images[0] * 255).type(torch.uint8), bbox, labels=labels, width=2)

    # target
    bbox_ref = targets[0]['boxes']
    labels = targets[0]['labels']
    bbox_ones = torch.ones_like(bbox_ref)
    bbox_ref = torch.where(bbox_ref > 1, bbox_ones, bbox_ref)
    bbox = bbox_ref.detach().clone()
    if len(bbox.shape) == 1:
        bbox = bbox.unsqueeze(dim=0)
        bbox_ref = bbox_ref.unsqueeze(dim=0)
    bbox[:, 0] = bbox_ref[:, 0] * (img_size[1] - 1)
    bbox[:, 1] = bbox_ref[:, 1] * (img_size[0] - 1)
    bbox[:, 2] = bbox_ref[:, 2] * (img_size[1] - 1)
    bbox[:, 3] = bbox_ref[:, 3] * (img_size[0] - 1)
    labels = torch.squeeze(labels).tolist()
    labels = [CLASS_NAMES[l] for l in labels] if isinstance(labels, list) else [CLASS_NAMES[labels]]
    # bbox = torch.unsqueeze(bbox[0], dim=0)
    # labels = [labels[0]]
    colours = ['red' for i in range(len(labels))]
    result = draw_bounding_boxes(result, bbox, labels=labels, width=2, colors=colours)

    show(result)
    plt.show()
    '''


if __name__ == '__main__':
    train(
        output_dir=r'E:\my_files\programmes\python\detector_output',
        session_name='retina-2cls',
        dataset_dir=r'E:\my_files\programmes\python\dp_dataset\full_dataset',
        img_size=(300, 400),
        resize_mode='stretch',
        epochs=200,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        model_type='retina',
        learning_rate=0.0001,
        trainable_backbone_layers=3,
    )
