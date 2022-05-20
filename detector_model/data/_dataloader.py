from typing import Iterator

import matplotlib.pyplot as plt
from torchvision.utils import draw_bounding_boxes
from torch.utils.data import DataLoader
import torchvision.transforms.functional as F
from collections import Iterator, Iterable

from voc_dataset import VOCDataset


class VOCDataLoader(Iterator):
    def __init__(
            self,
            dataset: VOCDataset,
            batch_size: int,
            shuffle: bool,
            num_workers: int,
            drop_last: bool,
    ):
        self.data_loader = DataLoader(
            dataset=dataset,
            shuffle=shuffle,
            num_workers=num_workers,
            batch_size=1,
        )
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.remains = self.data_loader.__len__()
        self.iterator = self.data_loader.__iter__()

    def reset(self):
        self.remains = self.data_loader.__len__()
        self.iterator = self.data_loader.__iter__()

    def __len__(self):
        return self.data_loader.__len__()

    def __next__(self):
        if self.remains >= self.batch_size:
            pass
