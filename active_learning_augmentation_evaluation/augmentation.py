import os
import cv2
import numpy
import numpy as np
import random
import xml.etree.ElementTree as ElementTree

import torch
from PIL import ImageEnhance
from torchvision import transforms
from PIL import Image
from skimage import exposure


def detector_augmentation(
        img: torch.Tensor,
        bboxes: torch.Tensor,
        labels: torch.Tensor,
        is_crop: bool,
        is_flip: bool,
        is_light: bool,
        is_contrast: bool,
        is_saturation: bool,
        is_hue: bool,
        is_sharp: bool,
        is_noise: bool,
        is_blur: bool,
):
    img = img.numpy()
    img = numpy.transpose(img, (1, 2, 0))
    img = Image.fromarray(img)
    bboxes = bboxes.numpy().tolist()
    labels = labels.numpy().tolist()

    width = float(img.size[0])
    height = float(img.size[1])
    if width > 800 or height > 800:
        img, bboxes = _change_size(img, bboxes, width, height)

    if is_crop:
        img, bboxes, labels = _crop_with_bbox(img, _my_uniform(3 / 5, 5 / 6), bboxes, labels)
        pass
    if is_flip:
        img, bboxes = _flip_horizontal_with_bbox(img, bboxes)
    if is_light:
        img = _change_light(img)
    if is_contrast:
        img = _change_contrast(img)
    if is_saturation:
        img = _change_saturation(img)
    if is_hue:
        img = _change_hue(img)
    if is_sharp:
        img = _change_sharp(img)
    if is_noise:
        img = _gaussian_noise(img, 0, 0.12)
    if is_blur:
        img = _gaussian_blur(img)

    img = torch.from_numpy(numpy.transpose(np.array(img), (2, 0, 1)))
    bboxes = torch.from_numpy(np.array(bboxes))
    labels = torch.from_numpy(np.array(labels))
    return img, bboxes, labels


def classifier_augmentation(
        img: torch.Tensor,
        is_crop: bool,
        is_flip: bool,
        is_light: bool,
        is_contrast: bool,
        is_saturation: bool,
        is_hue: bool,
        is_sharp: bool,
        is_noise: bool,
        is_blur: bool,
):
    img = img.numpy()
    # print(img.shape)
    img = numpy.transpose(img, (1, 2, 0))
    # print(img.shape)
    img = Image.fromarray(img, mode='RGB')

    if is_crop:
        img, bboxes, labels = _crop_without_bbox(img, _my_uniform(4 / 5, 5 / 6))
    if is_flip:
        img = _flip_horizontal_without_bbox(img)
    if is_light:
        img = _change_light(img)
    if is_contrast:
        img = _change_contrast(img)
    if is_saturation:
        img = _change_saturation(img)
    if is_hue:
        img = _change_hue(img)
    if is_sharp:
        img = _change_sharp(img)
    if is_noise:
        img = _gaussian_noise(img, 0, 0.12)
    if is_blur:
        img = _gaussian_blur(img)

    img = torch.from_numpy(numpy.transpose(np.array(img), (2, 0, 1)))
    # print(img.shape)
    return img


def _change_size(img, bboxes, width, height):
    base = 800
    if width >= height:
        precent = base / width
        width = base
        height = int(height * precent)
        img = img.resize((width, height), Image.ANTIALIAS)
    else:
        precent = base / height
        height = base
        width = int(width * precent)
        img = img.resize((width, height), Image.ANTIALIAS)

    size_bboxes = list()
    for bbox in bboxes:
        bbox[0] = int(bbox[0] * precent)
        bbox[1] = int(bbox[1] * precent)
        bbox[2] = int(bbox[2] * precent)
        bbox[3] = int(bbox[3] * precent)
        size_bboxes.append([bbox[0], bbox[1], bbox[2], bbox[3]])

    return img, size_bboxes


def _my_uniform(a, b):
    while True:
        result = random.uniform(a, b)
        if a < result < b:
            return result


def _flip_horizontal_with_bbox(img, bboxes):
    """
    Flip the image and box horizontally
    :param img: jpg/png image
    :param bboxes: Contains multiple sets of box location coordinates.
              Coordinates are composed of four numbers from the upper left and lower right coordinates.  
    """
    # flip horizontal img
    transform = transforms.RandomHorizontalFlip(p=1)
    img = transform(img)

    # flip horizontal bboxes 
    width = img.size[0]
    height = img.size[1]

    flip_bboxes = list()
    for bbox in bboxes:
        # flip_bboxes.append([width - bbox[0], bbox[1], width - bbox[2], bbox[3]])
        flip_bboxes.append([width - bbox[2], bbox[1], width - bbox[0], bbox[3]])

    return img, flip_bboxes


def _flip_horizontal_without_bbox(img):
    """
    Flip the image horizontally
    :param img: jpg/png image
    """
    # flip horizontal img
    transform = transforms.RandomHorizontalFlip(p=1)
    img = transform(img)
    return img


def _crop_without_bbox(img, crop_prop):
    """
    Capture an arbitrary proportion of an image
    Bboxes are reduced or deleted as they are clipped
    :param img: jpg/png image
    :param crop_prop: Cutting ratio
    """
    width = img.size[0]
    height = img.size[1]
    x_max = width
    x_min = 0
    y_max = height
    y_min = 0
    crop_width = int(crop_prop * x_max)
    crop_height = int(crop_prop * y_max)
    diff_width = width - crop_width
    diff_height = height - crop_height
    crop_x_min = int(random.uniform(0, diff_width))
    crop_y_min = int(random.uniform(0, diff_height))
    crop_x_max = crop_x_min + crop_width
    crop_y_max = crop_y_min + crop_height

    # Cut out the image
    crop_img = img.crop((crop_x_min, crop_y_min, crop_x_max, crop_y_max))
    return crop_img


def _crop_with_bbox(img, crop_prop, bboxes, labels):
    """
    Capture an arbitrary proportion of an image
    Bboxes are reduced or deleted as they are clipped
    :param img: jpg/png image
    :param crop_prop: Cutting ratio
    :param bboxes: Contains multiple sets of box location coordinates.
              Coordinates are composed of four numbers from the upper left and lower right coordinates.  
    :param labels: A list of number corresponding to the classes.
    """
    width = img.size[0]
    height = img.size[1]
    x_max = width
    x_min = 0
    y_max = height
    y_min = 0
    crop_width = int(crop_prop * x_max)
    crop_height = int(crop_prop * y_max)
    diff_width = width - crop_width
    diff_height = height - crop_height
    crop_x_min = int(random.uniform(0, diff_width))
    crop_y_min = int(random.uniform(0, diff_height))
    crop_x_max = crop_x_min + crop_width
    crop_y_max = crop_y_min + crop_height

    # Cut out the image
    crop_img = img.crop((crop_x_min, crop_y_min, crop_x_max, crop_y_max))

    # Cut out the bboxes
    crop_bboxes = list()
    crop_labels = list()
    for idx, bbox in enumerate(bboxes):
        bbox[0] = bbox[0] - crop_x_min
        bbox[1] = bbox[1] - crop_y_min
        bbox[2] = bbox[2] - crop_x_min
        bbox[3] = bbox[3] - crop_y_min

        for i in range(4):
            if bbox[i] < 0:
                bbox[i] = 0
            elif (i % 2) == 0 and bbox[i] > crop_x_max:
                bbox[i] = crop_x_max
            elif (i % 2) != 0 and bbox[i] > crop_y_max:
                bbox[i] = crop_y_max

        if bbox[0] == bbox[2] or bbox[1] == bbox[3]:
            continue

        crop_bboxes.append([bbox[0], bbox[1], bbox[2], bbox[3]])
        crop_labels.append(labels[idx])

    return crop_img, crop_bboxes, crop_labels


def _change_light(img):
    """
    The brightness will dim or brighten randomly
    """
    transform = transforms.ColorJitter(brightness=random.uniform(1/2, 4/5))
    img = transform(img)
    return img


def _change_contrast(img):
    """
    The contrast will change randomly within a certain range 80%~120% (1-0.2~1+0.2)
    """
    transform = transforms.ColorJitter(contrast=0.2)
    img = transform(img)
    return img


def _change_saturation(img):
    """
    The saturation will change randomly within a certain range 80%~120% (1-0.2~1+0.2)
    """
    transform = transforms.ColorJitter(saturation=0.2)
    img = transform(img)
    return img


def _change_hue(img):
    """
    The hue will change randomly within a certain range 80%~120% (1-0.2~1+0.2)
    """
    transform = transforms.ColorJitter(hue=0.2)
    img = transform(img)
    return img


def _change_sharp(img):
    """
    The sharp will change randomly within a certain range
    0 is fuzzy
    2 is Complete sharpening
    """
    enh_sha = ImageEnhance.Sharpness(img)
    new_img = enh_sha.enhance(factor=random.uniform(3/4, 4/3))
    return new_img


def _gaussian_noise(img, mean, sigma):
    """
    Add Gaussian noise to the image
    """
    # Standardize the image
    img = np.array(img)
    img = img / 255

    # Generate Gaussian noise
    G_noise = np.random.normal(mean, sigma, img.shape)

    # Overlay the noise with the image
    gaussian_img = img + G_noise
    gaussian_img = np.clip(gaussian_img, 0, 1)
    gaussian_img = np.uint8(gaussian_img * 255)
    gaussian_img = Image.fromarray(np.uint8(gaussian_img))
    return gaussian_img


def _gaussian_blur(img):
    """
    Gaussian blur the image
    """
    transform = transforms.GaussianBlur(21, 2)
    gaussian_img = transform(img)
    return gaussian_img


if __name__ == '__main__':
    img = Image.open('C:\\Users\\hp\\Desktop\\2019-04-2416-03-00.jpg')

    bboxes = ([[152, 156, 263, 261],
               [305, 25, 581, 384],
               [386, 27, 502, 175],
               [405, 95, 429, 111],
               [388, 126, 408, 146],
               [578, 96, 683, 333],
               [347, 103, 507, 373],
               [152, 198, 457, 384]])

    img_pre = detector_augmentation(img, bboxes, True, True, True, True, True, True, True, True, True)
    img_pre.show()
