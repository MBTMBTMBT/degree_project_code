import cv2
import torch
import numpy as np
import torchvision.transforms.transforms as transforms
import pickle
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn, fasterrcnn_resnet50_fpn
from torchvision.utils import draw_bounding_boxes
from data.voc_dataset import CLASS_NAMES

from my_utils.utils import load_model


class CameraDetector(object):
    def __init__(
            self,
            camera_index: int,
            img_size: tuple,
            model: torch.nn.Module,
            # resize_mode: str,
            show_window: bool,
    ):
        self.capture = cv2.VideoCapture(camera_index)
        self.camera_index = camera_index
        self.img_size = img_size
        self.transform = transforms.Resize(img_size, antialias=True)
        self.model = model
        self.show_window = show_window
        # self.resize_mode = resize_mode
        self.frame = None
        if not self.capture.isOpened():
            print("Cannot open camera, idx: %d" % camera_index)

    def read_frame(self):
        ret, frame = self.capture.read()
        if ret:
            # Our operations on the frame come here
            img_tensor = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_tensor = np.transpose(np.array(img_tensor), (2, 0, 1))
            img_tensor = torch.tensor(img_tensor).type(torch.float32) / 255
            self.frame = img_tensor
            # img_tensor = torch.unsqueeze(img_tensor, dim=0)
        else:
            print("Read frame failure, idx: %d" % self.camera_index)
            self.frame = None

    def detect(self):
        if self.frame is not None:
            img_tensor = torch.unsqueeze(self.frame, dim=0)
            img_tensor = self.transform(img_tensor)
            self.model.eval()
            prediction = self.model(img_tensor)
            # print(prediction)
            bbox_ref = prediction[0]['boxes']
            labels = prediction[0]['labels']
            scores = prediction[0]['scores']
            bbox_ones = torch.ones_like(bbox_ref)
            bbox_ref = torch.where(bbox_ref > 1, bbox_ones, bbox_ref)
            bbox = bbox_ref.detach().clone()
            if len(bbox.shape) == 1:
                bbox = bbox.unsqueeze(dim=0)
                bbox_ref = bbox_ref.unsqueeze(dim=0)
            bbox[:, 0] = bbox_ref[:, 0] * (self.img_size[1] - 1)
            bbox[:, 1] = bbox_ref[:, 1] * (self.img_size[0] - 1)
            bbox[:, 2] = bbox_ref[:, 2] * (self.img_size[1] - 1)
            bbox[:, 3] = bbox_ref[:, 3] * (self.img_size[0] - 1)
            labels = torch.squeeze(labels).tolist()
            labels = [CLASS_NAMES[l] for l in labels] if isinstance(labels, list) else [CLASS_NAMES[labels]]
            bbox = torch.unsqueeze(bbox[0], dim=0)
            labels = [labels[0]]
            if self.show_window:
                result = (img_tensor[0] * 255).type(torch.uint8)
                if scores[0] > 0.6:
                    result = draw_bounding_boxes(result, bbox, labels=labels, width=2)
                # result = draw_bounding_boxes(result, bbox, labels=labels, width=2)
                result = result.numpy()
                result = np.transpose(result, (1, 2, 0))
                result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
                cv2.imshow("frame", result)
        else:
            print("No frame read yet!")


if __name__ == '__main__':
    model = fasterrcnn_resnet50_fpn(
        pretrained=False,
        progress=True,
        num_classes=8,
        pretrained_backbone=False,
    )
    model = torch.nn.DataParallel(model)
    model = load_model(
        model,
        r'E:\my_files\programmes\python\detector_output\3-19-resNet-t5l\saved_checkpoints\generator_param_601.pkl',
        pickle,
        device=torch.device("cpu")
    )
    camera_detector = CameraDetector(
        camera_index=0,
        img_size=(300, 400),
        model=model,
        show_window=True,
    )
    while True:
        camera_detector.read_frame()
        camera_detector.detect()
        # Display the resulting frame
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
