import abc
from enum import Enum, unique
import torch
import numpy as np
import torchvision.transforms.transforms as transforms
from torchvision.utils import draw_bounding_boxes
from data.voc_dataset import CLASS_NAMES
import time
import mediapipe as mp
import cv2


CLASS_NAMES = (
    'Nan',
    'drvr_head',
    'drvr_hand',
    # 'seat_belt',
)

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


class MediapipeDetector(DetectorAbstract):

    def __init__(self):
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5,
            model_complexity=2
        )
        self.face = mp.solutions.face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.5
        )
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def detect(self, image_rgb: np.ndarray) -> DetectorRefBoundingBoxes:
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        # enum_landmarks = mp.solutions.pose.PoseLandmark

        results_faces = self.face.process(image_rgb)
        results_pose = self.pose.process(image_rgb)
        results_hands = self.hands.process(image_rgb)
        annotated_image = np.copy(image_rgb)

        # === get face rect ===
        scores = []
        ref_bboxes = []
        ref_sizes = []
        head_max_bbox_ref = (0, 0, 0, 0)
        if results_faces.detections:
            for detection in results_faces.detections:
                score = detection.score[0]
                ref_bounding_box_obj = detection.location_data.relative_bounding_box
                # print(ref_bounding_box_obj)
                ref_bounding_box = {
                    'xmin': ref_bounding_box_obj.xmin,
                    'ymin': ref_bounding_box_obj.ymin,
                    'width': ref_bounding_box_obj.width,
                    'height': ref_bounding_box_obj.height,
                }
                ref_bounding_box['xmax'] = ref_bounding_box['xmin'] + ref_bounding_box['width']
                ref_bounding_box['ymax'] = ref_bounding_box['ymin'] + ref_bounding_box['height']
                size = ref_bounding_box['width'] * ref_bounding_box['height']
                scores.append(score)
                ref_bboxes.append(ref_bounding_box)
                ref_sizes.append(size)
            max_size_arg = np.argmax(np.array(ref_sizes))
            max_bbox_ref_dict = ref_bboxes[max_size_arg]
            head_max_bbox_ref = [
                max_bbox_ref_dict['xmin'],
                max_bbox_ref_dict['ymin'],
                max_bbox_ref_dict['xmax'],
                max_bbox_ref_dict['ymax'],
            ]
            # head_max_bbox_ref[0] -= max_bbox_ref_dict['width'] / 6
            # head_max_bbox_ref[2] += max_bbox_ref_dict['width'] / 6
            head_max_bbox_ref[1] -= max_bbox_ref_dict['height'] / 3
            head_max_bbox_ref[3] += max_bbox_ref_dict['height'] / 8
            for i in range(len(head_max_bbox_ref)):
                if head_max_bbox_ref[i] > 1:
                    head_max_bbox_ref[i] = 1
                elif head_max_bbox_ref[i] < 0:
                    head_max_bbox_ref[i] = 0

        # collect face coordinates
        '''
        left_ear_x = results_pose.pose_landmarks.landmark[enum_landmarks.LEFT_EAR].x
        left_ear_y = results_pose.pose_landmarks.landmark[enum_landmarks.LEFT_EAR].y
        right_ear_x = results_pose.pose_landmarks.landmark[enum_landmarks.RIGHT_EAR].x
        right_ear_y = results_pose.pose_landmarks.landmark[enum_landmarks.RIGHT_EAR].y
        nose_x = results_pose.pose_landmarks.landmark[enum_landmarks.NOSE].x
        nose_y = results_pose.pose_landmarks.landmark[enum_landmarks.NOSE].y
        mouth_left_x = results_pose.pose_landmarks.landmark[enum_landmarks.MOUTH_LEFT].x
        mouth_left_y = results_pose.pose_landmarks.landmark[enum_landmarks.MOUTH_LEFT].y
        mouth_right_x = results_pose.pose_landmarks.landmark[enum_landmarks.MOUTH_RIGHT].x
        mouth_right_y = results_pose.pose_landmarks.landmark[enum_landmarks.MOUTH_RIGHT].y
        left_eye_x = results_pose.pose_landmarks.landmark[enum_landmarks.LEFT_EYE_OUTER].x
        left_eye_y = results_pose.pose_landmarks.landmark[enum_landmarks.LEFT_EYE_OUTER].y
        right_eye_x = results_pose.pose_landmarks.landmark[enum_landmarks.RIGHT_EYE_OUTER].x
        right_eye_y = results_pose.pose_landmarks.landmark[enum_landmarks.RIGHT_EYE_OUTER].y
        # collect face main coordinates
        mouth_y = max(mouth_left_y, mouth_right_y)  # get the lower mouth coordinate
        eyes_y = max(left_eye_y, right_eye_y)  # get the higher eye coordinate
        # collect face distances
        nose_mouth_vertical_distance = abs(nose_y - mouth_y)
        nose_eyes_vertical_distance = abs(nose_y - eyes_y)
        '''

        # === get hands rects ===
        # get hands from "hands" solution
        hands_list = []
        hands_bboxes = []
        if results_hands.multi_hand_landmarks:
            for handLms in results_hands.multi_hand_landmarks:
                mp_drawing.draw_landmarks(annotated_image, handLms, mp.solutions.hands.HAND_CONNECTIONS)
                hand_landmarks = mp.solutions.hands.HandLandmark
                hand_wrist_x = handLms.landmark[hand_landmarks.WRIST].x
                hand_wrist_y = handLms.landmark[hand_landmarks.WRIST].y
                hand_nodes_x = []
                hand_nodes_y = []
                for lm in handLms.landmark:
                    hand_nodes_x.append(lm.x)
                    hand_nodes_y.append(lm.y)
                hand = {
                    'wrist_x': hand_wrist_x,
                    'wrist_y': hand_wrist_y,
                    'nodes_x': hand_nodes_x,
                    'nodes_y': hand_nodes_y,
                }
                hands_list.append(hand)
                # find the largest and smallest xs and ys
                hand_x_min = min(hand_nodes_x)
                hand_y_min = min(hand_nodes_y)
                hand_x_max = max(hand_nodes_x)
                hand_y_max = max(hand_nodes_y)
                hand_box_width = hand_x_max - hand_x_min
                hand_box_height = hand_y_max - hand_y_min
                expand_x = hand_box_width / 8
                expand_y = hand_box_height / 8
                hand_x_min -= expand_x
                hand_y_min -= expand_y
                hand_x_max += expand_x
                hand_y_max += expand_y
                temp = np.array([hand_x_min, hand_y_min, hand_x_max, hand_y_max])
                temp = np.clip(temp, a_min=0, a_max=1)
                temp = temp.tolist()
                bbox = tuple(temp)
                hands_bboxes.append(bbox)
            # print(hands_list)

        if len(hands_bboxes) == 0:
            hand_a, hand_b = (), ()
        elif len(hands_bboxes) == 1:
            hand_a, hand_b = hands_bboxes[0], ()
        else:
            hand_a, hand_b = hands_bboxes[0], hands_bboxes[1]

        returned_rst = DetectorRefBoundingBoxes(head_max_bbox_ref, hand_a, hand_b)

        '''
        # get hands from "pose" solution
        left_wrist_x = results_pose.pose_landmarks.landmark[enum_landmarks.LEFT_WRIST].x
        left_wrist_y = results_pose.pose_landmarks.landmark[enum_landmarks.LEFT_WRIST].y
        right_wrist_x = results_pose.pose_landmarks.landmark[enum_landmarks.RIGHT_WRIST].x
        right_wrist_y = results_pose.pose_landmarks.landmark[enum_landmarks.RIGHT_WRIST].y
        '''

        '''
        # decide left hand and right hand
        left_hand, right_hand = None, None
        assert len(hands_list) <= 2
        if len(hands_list) == 1:
            hand = hands_list[0]
            left_distance = ((hand['wrist_x'] - left_wrist_x) ** 2 + (hand['wrist_y'] - left_wrist_y) ** 2) ** 0.5
            right_distance = ((hand['wrist_x'] - left_wrist_x) ** 2 + (hand['wrist_y'] - left_wrist_y) ** 2) ** 0.5
            if left_distance < right_distance:
                left_hand = 
        '''

        return returned_rst


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    detector = MediapipeDetector()
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
