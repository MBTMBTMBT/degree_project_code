import os.path

import numpy as np
import datetime
import cv2

from my_utils.detector import *
from my_utils.write_xml import *
from my_utils.utils import make_dir


class ImgLabeler(object):
    def __init__(self):
        self._detector = MediapipeDetector()

    def label(self, img_rgb: np.ndarray, out_img_dir: str, out_xml_dir: str, img_name='') -> DetectorRefBoundingBoxes:
        rst = self._detector.detect(img_rgb)
        make_dir(out_img_dir)
        make_dir(out_xml_dir)
        # time.sleep(0.001)  # sleep very short time to get different time stamp
        # current_time = img_name + str(datetime.datetime.now()).replace(' ', '-').replace(':', '-').replace('.', '-')
        tag_names = []
        bboxes = []
        if len(rst.get(BBoxComponents.HEAD)) > 0:
            tag_names.append('drvr_head')
            bboxes.append(rst.get(BBoxComponents.HEAD))
        if len(rst.get(BBoxComponents.HAND_A)) > 0:
            tag_names.append('drvr_hand')
            bboxes.append(rst.get(BBoxComponents.HAND_A))
        if len(rst.get(BBoxComponents.HAND_B)) > 0:
            tag_names.append('drvr_hand')
            bboxes.append(rst.get(BBoxComponents.HAND_B))
        write_xml(
            imgs_folder='imgs',
            img_name=img_name + '.png',
            img_path=os.path.join(out_img_dir, img_name + '.png'),
            xml_folder=out_xml_dir,
            img_width=img_rgb.shape[1],
            img_height=img_rgb.shape[0],
            tag_names=tag_names,
            boxes=bboxes,
            ref=True,
        )
        img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(out_img_dir, img_name + '.png'), img)
        return rst


class VideoLabeler(object):
    def __init__(self):
        self._labeler = ImgLabeler()

    def label(self, video_dir: str, out_img_dir: str, out_xml_dir: str, show_img=False, skip=10):
        for video_full_name in os.listdir(video_dir):
            video_path = os.path.join(video_dir, video_full_name)
            video_name = os.path.basename(video_path).split('.')[0]
            capture = cv2.VideoCapture(video_path)
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            for i in range(frame_count):
                ret, frame = capture.read()
                if (i + 1) % skip != 0:
                    continue
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    bbox_obj = self._labeler.label(frame, out_img_dir, out_xml_dir, img_name=video_name + '-' + str(i))
                    if show_img:
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        bbox_obj.draw(frame)
                        cv2.imshow('frame', frame)
                        cv2.waitKey(1)
            capture.release()
        cv2.destroyAllWindows()


class ImgPatchLabeler(object):
    def __init__(self):
        self._labeler = ImgLabeler()

    def label(self, imgs_dir: str, out_img_dir: str, out_xml_dir: str, show_img=False, skip=10):
        for img_full_name in os.listdir(imgs_dir):
            try:
                if img_full_name.split('.')[-1] == 'jpg' \
                        or img_full_name.split('.')[-1] == 'JPG' \
                        or img_full_name.split('.')[-1] == 'png':
                    img_path = os.path.join(imgs_dir, img_full_name)
                    img_name = os.path.basename(img_path).split('.')[0]
                    img = cv2.imread(img_path)
                    if img.shape[0] > 2048 and img.shape[1] > 2048:
                        img = cv2.resize(img, (int(img.shape[1] / 4), int(img.shape[0] / 4)), interpolation=cv2.INTER_AREA)
                    elif img.shape[0] > 1024 and img.shape[1] > 1024:
                        img = cv2.resize(img, (int(img.shape[1] / 2), int(img.shape[0] / 2)),
                                         interpolation=cv2.INTER_AREA)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    bbox_obj = self._labeler.label(img, out_img_dir, out_xml_dir, img_name=img_name)
                    if show_img:
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        bbox_obj.draw(img)
                        cv2.imshow('img', img)
                        cv2.waitKey(1)
            except Exception as e:
                print(e)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    # '''
    video_lblr = VideoLabeler()
    video_lblr.label(
        r'E:\my_files\programmes\python\dp_dataset\full_dataset\new_videos',
        r'E:\my_files\programmes\python\dp_dataset\full_dataset\imgs',
        r'E:\my_files\programmes\python\dp_dataset\full_dataset\annotations',
        show_img=True,
        skip=10,
    )
    '''
    img_lblr = ImgPatchLabeler()
    img_lblr.label(
        r'E:\my_files\programmes\python\dp_dataset\full_dataset\new',
        r'E:\my_files\programmes\python\dp_dataset\full_dataset\imgs',
        r'E:\my_files\programmes\python\dp_dataset\full_dataset\annotations',
        show_img=True,
        skip=10,
    )
    '''
