import cv2
import numpy as np
import pickle
import torch
import os

from torchvision.models.detection import fasterrcnn_resnet50_fpn

from detector import DetectorRefBoundingBoxes, BoundingBoxDetector, \
    DetectorAbstract, BBoxComponents
from classifier import Classifier
from my_utils import load_model, load_model_cancel_parallel
from classifier_model import ClassifierModel
from classes import *


class DetectorClassifier(object):
    def __init__(
            self,
            detector: DetectorAbstract,
            head_motion_classifier: Classifier,
            hand_motion_classifier: Classifier,
            sunglasses_classifier: Classifier,
            head_mask_classifier: Classifier,
    ):
        self.detector = detector
        self.head_motion_classifier = head_motion_classifier
        self.sunglasses_classifier = sunglasses_classifier
        self.head_mask_classifier = head_mask_classifier
        self.hand_motion_classifier = hand_motion_classifier

    @staticmethod
    def _cv2_img_preprocess(img: np.ndarray) -> np.ndarray:
        img = img.transpose((2, 0, 1))
        img = img.astype(np.float32)
        img /= 255
        img = np.clip(img, a_min=0, a_max=1)
        return img

    @staticmethod
    def _crop_channel_first_image(img: np.ndarray, bbox_ref: tuple):
        bbox = [0, 0, 0, 0]
        bbox[0] = int(bbox_ref[0] * img.shape[2])
        bbox[1] = int(bbox_ref[1] * img.shape[1])
        bbox[2] = int(bbox_ref[2] * img.shape[2])
        bbox[3] = int(bbox_ref[3] * img.shape[1])
        img_cropped = img[:, bbox[1]:bbox[3], bbox[0]:bbox[2]]
        return img_cropped

    def detect_and_classify(
            self,
            rgb_image: np.ndarray,
    ):
        """

        :param rgb_image: image come directly from OpenCV.
        :return:
        """
        img_copy = rgb_image.copy()
        # rgb_image = DetectorClassifier._cv2_img_preprocess(rgb_image)
        predicted_bounding_boxes = self.detector.detect(rgb_image)
        head_bbox = predicted_bounding_boxes.get(BBoxComponents.HEAD)
        hand_bbox = predicted_bounding_boxes.get(BBoxComponents.HAND_A)
        if sum(head_bbox) == 0:
            return
        rgb_image = DetectorClassifier._cv2_img_preprocess(rgb_image)
        # print(rgb_image.shape)
        cropped_head = DetectorClassifier._crop_channel_first_image(rgb_image, head_bbox)
        cropped_head = torch.tensor(cropped_head)
        cropped_hand = DetectorClassifier._crop_channel_first_image(rgb_image, hand_bbox)
        cropped_hand = torch.tensor(cropped_hand)
        # print(cropped_head.shape)
        _, head_classify_rst = self.head_motion_classifier.classify(cropped_head)
        _, hand_class_rst = self.hand_motion_classifier.classify(cropped_hand)
        _, sunglasses_classify_rst = self.sunglasses_classifier.classify(cropped_head)
        _, mask_classify_rst = self.head_mask_classifier.classify(cropped_head)
        # print(head_classify_rst)
        head_class = HEAD_MOTIONS[head_classify_rst]
        hand_class = HAND_MOTIONS[hand_class_rst]
        sunglasses = SUNGLASSES_CLASSES[sunglasses_classify_rst]
        mask = MASK_CLASSES[mask_classify_rst]
        print(head_class)
        print('sunglasses', sunglasses)
        print('mask', mask)
        predicted_bounding_boxes.draw(img_copy)
        # img_copy = cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR)
        # cv2.imshow('frame', img_copy)
        # cv2.waitKey(1)
        if head_class == 'looking_away' or head_class == 'talking_on_the_phone' or head_class == 'eating_or_drinking':
            os.system("beep.mp4")
        elif hand_class == 'food, drink and cigarette' or hand_class == 'mobile phone':
            os.system("beep.mp4")
        return head_class, sunglasses, mask


if __name__ == '__main__':
    capture = cv2.VideoCapture(0)
    detector_model = fasterrcnn_resnet50_fpn(
        pretrained=False,
        progress=True,
        num_classes=3,
        pretrained_backbone=False,
        trainable_backbone_layers=5,
    )
    # model = torch.nn.DataParallel(model)
    detector_model = load_model(
        detector_model,
        r'models/detector.pkl',
        pickle,
        device=torch.device("cpu")
    )
    detector = BoundingBoxDetector(img_size=(300, 400), model=detector_model)
    head_classifier_model = ClassifierModel('resNet', len(HEAD_MOTIONS))
    head_classifier_model = load_model_cancel_parallel(
        head_classifier_model,
        r'models/head_classifier.pkl',
        pickle,
        device=torch.device("cpu")
    )
    head_motion_classifier = Classifier(
        img_size=(120, 120),
        model=head_classifier_model,
        show_window=True,
        class_list=HEAD_MOTIONS,
    )
    sunglasses_classifier = ClassifierModel('mobileNet', len(SUNGLASSES_CLASSES))
    sunglasses_classifier = load_model_cancel_parallel(
        sunglasses_classifier,
        r'models/sunglasses_classifier.pkl',
        pickle,
        device=torch.device("cpu")
    )
    sunglasses_classifier = Classifier(
        img_size=(120, 120),
        model=sunglasses_classifier,
        show_window=True,
        class_list=SUNGLASSES_CLASSES,
    )
    head_mask_classifier = ClassifierModel('mobileNet', len(MASK_CLASSES))
    head_mask_classifier = load_model_cancel_parallel(
        head_mask_classifier,
        r'models/mask_classifier.pkl',
        pickle,
        device=torch.device("cpu")
    )
    head_mask_classifier = Classifier(
        img_size=(120, 120),
        model=head_mask_classifier,
        show_window=True,
        class_list=MASK_CLASSES,
    )
    # detector_classifier = DetectorClassifier(
    #     detector=detector,
    #     head_motion_classifier=head_motion_classifier,
    #     sunglasses_classifier=sunglasses_classifier,
    #     head_mask_classifier=head_mask_classifier,
    # )
    # while True:
    #     ret, frame = capture.read()
    #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #     detector_classifier.detect_and_classify(frame)
