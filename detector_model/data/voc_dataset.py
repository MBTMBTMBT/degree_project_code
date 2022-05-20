import xml.etree.ElementTree as ElementTree
# from PIL import Image
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import os
import operator
import torchvision.transforms.transforms as transforms
import random

import my_utils.utils
try:
    from data.augmentation import detector_augmentation
except ModuleNotFoundError:
    from augmentation import detector_augmentation

'''
CLASS_NAMES = (
    'Nan',
    'driver',
    'drvr_head',
    'drvr_hand',
    'drvr_eyes',
    'drvr_sun_glasses',
    'drvr_mouth',
    'drvr_mask',
    'passenger',
    'steering_wheel',
    'seat_belt',
    'mobile_phone',
    'water',
    'food',
    'cigarette',
)
'''

'''
CLASS_NAMES = (
    'Nan',
    'drvr_head',
    'drvr_hand',
    'seat_belt',
    'mobile_phone',
    'water_or_food',
    'cigarette',
)
'''

CLASS_NAMES = (
    'Nan',
    'drvr_head',
    'drvr_hand',
    # 'seat_belt',
)

'''
CLASS_NAMES = (
    'Nan',
    'drvr_hand',
)
'''

'''
CLASS_NAMES_MAP = {
    'Nan': 'Nan',
    'drvr_head': 'drvr_head',
    'drvr_hand': 'drvr_hand',
    'seat_belt': 'seat_belt',
    'mobile_phone': 'mobile_phone',
    'water': 'water_or_food',
    'food': 'water_or_food',
    'cigarette': 'cigarette',
}
'''

CLASS_NAMES_MAP = {
    'Nan': 'Nan',
    'drvr_head': 'drvr_head',
    'drvr_hand': 'drvr_hand',
    # 'seat_belt': 'seat_belt',
}

'''
CLASS_NAMES_MAP = {
    'Nan': 'Nan',
    'drvr_hand': 'drvr_hand',
}
'''

RESIZE_MODES = (
    'stretch',
    'pad',
)


class VOCDataset(Dataset):
    def __init__(
            self,
            dataset_dir: str,
            size: tuple,
            resize_mode: str,
            use_random_augmentation: bool,
            return_filename=False,
    ):
        """
        initialize the VOCDataset
        :param dataset_dir: the root directory of the VOC dataset,
            inside it should contain two folders: imgs and annotations.
        :param size: the expected output size of the image
        :param resize_mode: select from 'stretch' and 'pad',
            'stretch': output will be directly scaled into the expected shape.
            'pad': output will be first scaled and then padded with zero into the expected shape.
        :param use_random_augmentation: use or not, data augmentation
        """
        super().__init__()
        self.dataset_dir = dataset_dir
        self.size = size
        self.transform = transforms.Resize(size, antialias=True)
        self.resize_mode = resize_mode
        self.num_classes = len(CLASS_NAMES)
        self.augmentation = use_random_augmentation
        self.return_filename = return_filename

        # prepare lists for images and annotations
        self.img_path_list = list()
        self.anno_path_list = list()
        img_dir = os.path.join(dataset_dir, 'imgs')
        anno_dir = os.path.join(dataset_dir, 'annotations')
        # check = os.listdir(img_dir)
        for each_img in os.listdir(img_dir):
            if each_img.split('.')[-1] == 'jpg' \
                    or each_img.split('.')[-1] == 'JPG' \
                    or each_img.split('.')[-1] == 'png':  # or each_img.split('.')[-1] == 'webp':
                img_path = os.path.join(img_dir, each_img)
                anno_name = ''
                for idx, each in enumerate(each_img.split('.')):
                    if idx < len(each_img.split('.')) - 1:
                        anno_name += each
                anno_name += '.xml'
                anno_path = os.path.join(anno_dir, anno_name)
                if os.path.isfile(anno_path):
                    self.img_path_list.append(img_path)
                    self.anno_path_list.append(anno_path)
            else:
                print("Exception: ", each_img)
        anno_count = 0
        for each_anno in os.listdir(anno_dir):
            if each_anno.split('.')[-1] == 'xml':
                anno_count += 1
        # assert anno_count == len(self.anno_path_list)

    def join(self, dataset):
        """
        merge two datasets
        :param dataset: another VOCDataset instance
        :return: None
        """
        assert operator.eq(self.size, dataset.size)
        self.img_path_list += dataset.img_path_list
        self.anno_path_list += dataset.anno_path_list

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        anno_path = self.anno_path_list[index]
        img = VOCDataset._read_image(img_path)
        # plt.imshow(img.permute((1, 2, 0)))
        bbox, label = VOCDataset._read_annotation(anno_path)

        if self.augmentation:
            img, bbox, label = detector_augmentation(
                img,
                bbox,
                label,
                bool(random.getrandbits(1)),
                bool(random.getrandbits(1)),
                bool(random.getrandbits(1)),
                False,
                False,
                False,
                False,
                False,
                False,
                # bool(random.getrandbits(1)),
                # bool(random.getrandbits(1)),
                # bool(random.getrandbits(1)),
                # bool(random.getrandbits(1)),
                # bool(random.getrandbits(1)),
                # bool(random.getrandbits(1)),
            )

        # img: (C, H, W)
        bbox_ref = torch.zeros(*bbox.shape)
        # print(bbox.shape)
        if bbox.shape[0] == 0:  # IN CASE SOME ONE LABELS THINGS WRONG!
            bbox_ref = torch.tensor([[0, 0, 0.01, 0.01]])
            label = torch.tensor([0])
            img, bbox_ref = self._resize(img, bbox_ref)
            # change values into 0 ~ 1
            img = img.type(torch.float32)
            img /= 255
            label = label.type(torch.int64)
            if self.return_filename:
                img_name = os.path.basename(img_path)
                return img, bbox_ref, label, img_name
            return img, bbox_ref, label

        if len(bbox.shape) == 1:
            bbox = bbox.unsqueeze(dim=0)
            bbox_ref = bbox_ref.unsqueeze(dim=0)
        bbox_ref[:, 0] = bbox[:, 0] / img.shape[2]
        bbox_ref[:, 1] = bbox[:, 1] / img.shape[1]
        bbox_ref[:, 2] = bbox[:, 2] / img.shape[2]
        bbox_ref[:, 3] = bbox[:, 3] / img.shape[1]
        # print(img_path)
        # print("ori")
        # print(img.shape)
        # print(bbox)
        # print(bbox_ref)

        # print(img.shape)
        img, bbox_ref = self._resize(img, bbox_ref)

        # change values into 0 ~ 1
        img = img.type(torch.float32)
        img /= 255
        label = label.type(torch.int64)

        if self.return_filename:
            img_name = os.path.basename(img_path)
            return img, bbox_ref, label, img_name

        return img, bbox_ref, label

    def _resize(self, img: torch.Tensor, bbox_ref: torch.Tensor) -> tuple:
        assert self.resize_mode in RESIZE_MODES
        if self.resize_mode == 'stretch':
            img = self.transform(img)
        if self.resize_mode == 'pad':
            if img.shape[1] / img.shape[2] <= self.size[0] / self.size[1]:
                # pad H
                padded_height = int(img.shape[2] * self.size[0] / self.size[1])  # W * h/w
                pad = torch.zeros(img.shape[0], padded_height, img.shape[2])
                side = int((padded_height - img.shape[1]) / 2)
                pad[:, side:side + img.shape[1], :] = img
                bbox = torch.zeros_like(bbox_ref)
                # print(pad.shape[1])
                bbox[:, 0] = bbox_ref[:, 0] * pad.shape[1]  # xmin
                bbox[:, 1] = bbox_ref[:, 1] * pad.shape[2]  # ymin
                bbox[:, 2] = bbox_ref[:, 2] * pad.shape[1]  # xmax
                bbox[:, 3] = bbox_ref[:, 3] * pad.shape[2]  # ymax
                bbox[:, 0] += side
                bbox[:, 2] += side
            else:
                # pad W
                padded_width = int(img.shape[1] * self.size[1] / self.size[0])  # H * w/h
                pad = torch.zeros(img.shape[0], img.shape[1], padded_width)
                side = (padded_width - img.shape[2]) // 2
                pad[:, :, side:side + img.shape[2]] = img
                bbox = (
                    bbox_ref[:, 0] * pad.shape[1],  # xmin
                    bbox_ref[:, 1] * pad.shape[2],  # ymin
                    bbox_ref[:, 2] * pad.shape[1],  # xmax
                    bbox_ref[:, 3] * pad.shape[2],  # ymax
                )
                bbox[1] += side
                bbox[3] += side
            bbox_ref[:, 0] = bbox[:, 0] / pad.shape[1]
            bbox_ref[:, 1] = bbox[:, 1] / pad.shape[2]
            bbox_ref[:, 2] = bbox[:, 2] / pad.shape[1]
            bbox_ref[:, 3] = bbox[:, 3] / pad.shape[2]
            img = self.transform(img)
        return img, bbox_ref

    @staticmethod
    def _read_annotation(anno_path: str) -> tuple:
        """
        Args:
            anno_path (str): Full path of the annotation.

        Returns:
            tuple of an image and bounding boxes
        """
        anno = ElementTree.parse(anno_path)
        bbox = list()
        label = list()
        for obj in anno.findall('object'):
            bndbox_anno = obj.find('bndbox')
            name = obj.find('name').text.lower().strip()
            if name in CLASS_NAMES_MAP.keys():
                # subtract 1 to make pixel indexes 0-based
                bbox.append([
                    int(bndbox_anno.find(tag).text) - 1
                    for tag in ('xmin', 'ymin', 'xmax', 'ymax')
                ])
                # print(CLASS_NAMES_MAP[name])
                label.append(CLASS_NAMES.index(CLASS_NAMES_MAP[name]))
        if len(bbox) != 0:
            bbox = np.stack(bbox).astype(np.float32)
            label = np.stack(label).astype(np.int32)
        else:
            bbox = np.array([])
            label = np.array([])
        bbox = torch.from_numpy(bbox)
        label = torch.from_numpy(label)
        return bbox, label

    @staticmethod
    def _read_image(img_path) -> torch.Tensor:
        """
        Args:
            img_path (str): Full path of the image.

        Returns:
            torch Tensor of the image
        """
        img = VOCDataset.__read_image(img_path)
        img = torch.from_numpy(img)
        return img

    @staticmethod
    def __read_image(img_path) -> np.ndarray:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # PIL sometimes can get up-side-down images, I hate it, so, lets use OpenCV~
        # img = Image.open(img_path)
        # img = f.convert('RGB')
        # img = np.array(img)
        if len(img.shape) == 2:
            # reshape (H, W) -> (1, H, W)
            return img[np.newaxis]
        else:
            # transpose (H, W, C) -> (C, H, W)
            return img.transpose((2, 0, 1))


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from torchvision.utils import draw_bounding_boxes
    from torch.utils.data import DataLoader

    test_dataset = VOCDataset(
        dataset_dir=r'E:\my_files\programmes\python\dp_dataset\full_dataset',
        size=(300, 400),
        resize_mode='stretch',
        use_random_augmentation=True,
    )
    test_loader = DataLoader(test_dataset, shuffle=True, num_workers=0)
    for img, bbox_ref, target in test_loader:
        bbox_ref = torch.squeeze(bbox_ref)
        bbox = torch.zeros(*bbox_ref.shape)
        if len(bbox.shape) == 1:
            bbox = bbox.unsqueeze(dim=0)
            bbox_ref = bbox_ref.unsqueeze(dim=0)
        bbox[:, 0] = bbox_ref[:, 0] * 400
        bbox[:, 1] = bbox_ref[:, 1] * 300
        bbox[:, 2] = bbox_ref[:, 2] * 400
        bbox[:, 3] = bbox_ref[:, 3] * 300
        # print("out")
        # print(bbox)
        labels = torch.squeeze(target).tolist()
        print(labels)
        labels = [CLASS_NAMES[l] for l in labels] if isinstance(labels, list) else [CLASS_NAMES[labels]]
        print(labels)
        # print(bbox_ref)
        img = torch.squeeze(img)
        img = draw_bounding_boxes((img * 255).type(torch.uint8), bbox, labels=labels, width=2)
        my_utils.utils.show(img)
        plt.show()
