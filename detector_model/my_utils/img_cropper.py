import os.path

import cv2
from torch.utils.data import DataLoader
import tqdm
import data.voc_dataset
import torch
import numpy as np
import torchvision.transforms.transforms as transforms
import pickle
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, fasterrcnn_resnet50_fpn
from torchvision.utils import draw_bounding_boxes
from data.voc_dataset import CLASS_NAMES
import matplotlib.pyplot as plt
from PIL import Image

from my_utils.utils import load_model, make_dir


class ImgCropper(object):
    def __init__(
            self,
            dataset_dir: str,
            img_size: tuple,
            resize_mode: str,
            class_list: list,
            output_dir: str,
    ):
        self.dataset = data.voc_dataset.VOCDataset(
            dataset_dir=dataset_dir,
            size=img_size,
            resize_mode=resize_mode,
            use_random_augmentation=False,
            return_filename=True,
        )
        self.dataloader = DataLoader(
            dataset=self.dataset,
            shuffle=False,
            num_workers=0,
            batch_size=1,
        )
        self.class_list = class_list
        self.output_dir = output_dir
        self.img_size = img_size
        make_dir(output_dir)

    def crop(self, cropped_class: str):
        class_label = self.class_list.index(cropped_class)
        for img, bbox_refs, labels, img_name in tqdm.tqdm(self.dataloader):
            img = torch.squeeze(img)
            bbox_refs = torch.squeeze(bbox_refs)
            labels = torch.squeeze(labels)
            img = img.numpy()
            img = np.transpose(img, (1, 2, 0))
            label_idxs = []
            try:
                for idx, label in enumerate(labels):
                    if label == class_label:
                        label_idxs.append(idx)
                bbox_refs_cropped = []
                for idx in label_idxs:
                    bbox_refs_cropped.append(bbox_refs[idx].numpy().tolist())
                for idx, bbox_ref in enumerate(bbox_refs_cropped):
                    bbox = np.zeros_like(bbox_ref)
                    bbox[0] = bbox_ref[0] * self.img_size[1]
                    bbox[1] = bbox_ref[1] * self.img_size[0]
                    bbox[2] = bbox_ref[2] * self.img_size[1]
                    bbox[3] = bbox_ref[3] * self.img_size[0]
                    bbox = bbox.astype('int')
                    img_cropped = img[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
                    img_cropped *= 255
                    img_cropped = img_cropped.astype('uint8')
                    # print(img_name)
                    save_name = img_name[0].split('.')[0] + cropped_class + str(idx) + '.' + img_name[0].split('.')[-1]
                    # print(save_name)
                    save_path = os.path.join(self.output_dir, save_name)
                    img_cropped = Image.fromarray(img_cropped)
                    img_cropped.save(save_path)
                    # print(save_path)
                    # plt.imshow(img_cropped)
                    # plt.show()
            except Exception as e:
                pass
                # print(e)


if __name__ == '__main__':
    img_cropper = ImgCropper(
        dataset_dir=r'E:\my_files\programmes\python\dp_dataset\full_dataset',
        img_size=(600, 800),
        resize_mode='stretch',
        class_list=data.voc_dataset.CLASS_NAMES,
        output_dir=r'E:\my_files\programmes\python\detector_output\crop_head',
    )
    img_cropper.crop('drvr_head')
