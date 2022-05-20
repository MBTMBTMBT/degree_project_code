import abc
from enum import Enum, unique
import torch
import numpy as np
import torchvision.transforms.transforms as transforms
from classes import CLASS_NAMES
import time
import cv2


@unique
class BBoxComponents(Enum):
    HEAD = 0
    HAND_A = 1
    HAND_B = 2


class DetectorRefBoundingBoxes(object):
    """
    The coordinates in this will only be between 0 and 1
    """

    def __init__(self, head: tuple, hand_a: tuple, hand_b: tuple):
        self._bbox_dict = {
            BBoxComponents.HEAD: head,
            BBoxComponents.HAND_A: hand_a,
            BBoxComponents.HAND_B: hand_b,
        }

    def get(self, component: BBoxComponents) -> tuple:
        """
        get the bounding box coordinates in tuples
        :param component: BBoxComponents
        :return:
        """
        return self._bbox_dict[component]

    def draw(self, img: np.ndarray):
        annotated_image = img
        img_shape = annotated_image.shape
        if len(self._bbox_dict[BBoxComponents.HEAD]) > 0 and sum(self._bbox_dict[BBoxComponents.HEAD]) > 0:
            cv2.rectangle(
                annotated_image,
                (
                    int(self._bbox_dict[BBoxComponents.HEAD][0] * img_shape[1]),
                    int(self._bbox_dict[BBoxComponents.HEAD][1] * img_shape[0])
                ),
                (
                    int(self._bbox_dict[BBoxComponents.HEAD][2] * img_shape[1]),
                    int(self._bbox_dict[BBoxComponents.HEAD][3] * img_shape[0])
                ),
                (0, 255, 0),
                2
            )
        if len(self._bbox_dict[BBoxComponents.HAND_A]) > 0 and sum(self._bbox_dict[BBoxComponents.HAND_A]) > 0:
            cv2.rectangle(
                annotated_image,
                (
                    int(self._bbox_dict[BBoxComponents.HAND_A][0] * img_shape[1]),
                    int(self._bbox_dict[BBoxComponents.HAND_A][1] * img_shape[0])
                ),
                (
                    int(self._bbox_dict[BBoxComponents.HAND_A][2] * img_shape[1]),
                    int(self._bbox_dict[BBoxComponents.HAND_A][3] * img_shape[0])
                ),
                (0, 255, 0),
                2
            )
        if len(self._bbox_dict[BBoxComponents.HAND_B]) > 0 and sum(self._bbox_dict[BBoxComponents.HAND_B]) > 0:
            cv2.rectangle(
                annotated_image,
                (
                    int(self._bbox_dict[BBoxComponents.HAND_B][0] * img_shape[1]),
                    int(self._bbox_dict[BBoxComponents.HAND_B][1] * img_shape[0])
                ),
                (
                    int(self._bbox_dict[BBoxComponents.HAND_B][2] * img_shape[1]),
                    int(self._bbox_dict[BBoxComponents.HAND_B][3] * img_shape[0])
                ),
                (0, 255, 0),
                2
            )
        # return annotated_image


class DetectorAbstract(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def detect(self, image_rgb: np.ndarray) -> DetectorRefBoundingBoxes:
        pass


class BoundingBoxDetector(DetectorAbstract):
    def __init__(
            self,
            img_size: tuple,
            model: torch.nn.Module,
    ):
        self.img_size = img_size
        self.transform = transforms.Resize(img_size, antialias=True)
        self.model = model

    def detect(self, image_rgb: np.ndarray) -> DetectorRefBoundingBoxes:
        img_tensor = torch.tensor(image_rgb)
        img_tensor = torch.unsqueeze(img_tensor, dim=0)
        img_tensor = self.transform(img_tensor)
        self.model.eval()
        prediction = self.model(img_tensor)
        # print(prediction)
        bbox_ref = prediction[0]['boxes']
        labels = prediction[0]['labels']
        scores = prediction[0]['scores']
        bbox_ones = torch.ones_like(bbox_ref)
        bbox_ref = torch.where(bbox_ref > 1, bbox_ones, bbox_ref)
        bbox_copy = bbox_ref.detach().clone()
        bbox_ref = bbox_ref.detach().clone()
        heads, hands = [], []
        head_bbox = (0, 0, 0, 0)
        for idx, each_label in enumerate(labels):
            if CLASS_NAMES[each_label] == 'drvr_head':
                head_bbox = bbox_ref[idx]
                print(each_label)
                print(scores[idx])
                break
        # todo: finish this!
        rst = DetectorRefBoundingBoxes(head_bbox, (), ())
        return rst


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    detector = BoundingBoxDetector((400, 300))
    pTime = 0
    cTime = 0
    while True:
        success, img = cap.read()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        annotated_image = img.copy()
        rst = detector.detect(imgRGB)
        rst.draw(annotated_image)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(annotated_image, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.imshow("Image", annotated_image)
        cv2.waitKey(1)
