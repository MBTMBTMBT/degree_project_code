import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import os
import operator
import torchvision.transforms.transforms as transforms
import random

try:
    from data.augmentation import classifier_augmentation
except ModuleNotFoundError:
    from augmentation import classifier_augmentation

MASK_CLASSES = (
    'not_wearing',
    'wearing',
)

SUNGLASSES_CLASSES = (
    'not_wearing',
    'wearing',
)

HEAD_MOTIONS = (
    'no_motion',
    'looking_away',
    'eating_or_drinking',
    'talking_on_the_phone',
    'smoking',
    'touch_face_or_head',
    'tired',
)

RESIZE_MODES = (
    'stretch',
    'pad',
)


class MultiClassDataset(Dataset):
    def __init__(
            self,
            dataset_dir: str,
            classes: tuple,
            size: tuple,
            resize_mode: str,
            use_random_augmentation: bool,
            return_filename=False,
    ):
        """
        initialize the MultiClassDataset
        :param dataset_dir: the root directory of the dataset,
            inside it should contain several folders, with the names of the classes.
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
        self.classes = classes
        self.num_classes = len(classes)
        self.augmentation = use_random_augmentation
        self.return_filename = return_filename

        # prepare lists for images and annotations
        self.img_path_list = list()
        self.label_list = list()
        for each_class in self.classes:
            img_dir = os.path.join(dataset_dir, each_class)
            for each_img in os.listdir(img_dir):
                if each_img.split('.')[-1] == 'jpg' \
                        or each_img.split('.')[-1] == 'JPG' \
                        or each_img.split('.')[-1] == 'png':  # or each_img.split('.')[-1] == 'webp':
                    img_path = os.path.join(img_dir, each_img)
                    self.img_path_list.append(img_path)
                    self.label_list.append(self.classes.index(each_class))
                else:
                    print("Exception: ", each_img)

    def join(self, dataset):
        """
        merge two datasets
        :param dataset: another VOCDataset instance
        :return: None
        """
        assert operator.eq(self.size, dataset.size)
        self.img_path_list += dataset.img_path_list

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        label = self.label_list[index]
        img = MultiClassDataset._read_image(img_path)
        # print(img.shape)

        if self.augmentation:
            img = classifier_augmentation(
                img,
                False,
                bool(random.getrandbits(1)),
                bool(random.getrandbits(1)),
                bool(random.getrandbits(1)),
                bool(random.getrandbits(1)),
                bool(random.getrandbits(1)),
                bool(random.getrandbits(1)),
                bool(random.getrandbits(1)),
                bool(random.getrandbits(1)),
            )

        # print(img.shape)
        img = self._resize(img,)

        # change values into 0 ~ 1
        img = img.type(torch.float32)
        img /= 255

        if self.return_filename:
            img_name = os.path.basename(img_path)
            return img, label, img_name

        return img, label

    def _resize(self, img: torch.Tensor,) -> tuple:
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
            else:
                # pad W
                padded_width = int(img.shape[1] * self.size[1] / self.size[0])  # H * w/h
                pad = torch.zeros(img.shape[0], img.shape[1], padded_width)
                side = (padded_width - img.shape[2]) // 2
                pad[:, :, side:side + img.shape[2]] = img
            img = self.transform(img)
        return img

    @staticmethod
    def _read_image(img_path) -> torch.Tensor:
        """
        Args:
            img_path (str): Full path of the image.

        Returns:
            torch Tensor of the image
        """
        img = MultiClassDataset.__read_image(img_path)
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

    test_dataset = MultiClassDataset(
        dataset_dir=r'E:\my_files\programmes\python\dp_dataset\mask',
        classes=MASK_CLASSES,
        size=(120, 120),
        resize_mode='stretch',
        use_random_augmentation=True,
    )
    test_loader = DataLoader(test_dataset, shuffle=True, num_workers=0, batch_size=4)
    for imgs, labels in test_loader:
        labels = torch.squeeze(labels)
        img = torch.squeeze(imgs)
        print(labels.shape)
        print(imgs.shape)
        # labels = [CLASS_NAMES[l] for l in labels] if isinstance(labels, list) else [CLASS_NAMES[labels]]
        # print(labels)
        # print(bbox_ref)
        # img = draw_bounding_boxes((img * 255).type(torch.uint8), bbox, labels=labels, width=2)
        # my_utils.utils.show(img)
        # plt.show()
