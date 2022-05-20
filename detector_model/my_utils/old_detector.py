import torch
import numpy as np
import torchvision.transforms.transforms as transforms
import pickle
from torchvision.models.detection import fasterrcnn_resnet50_fpn, fasterrcnn_mobilenet_v3_large_fpn, retinanet_resnet50_fpn
from torchvision.utils import draw_bounding_boxes
from data.voc_dataset import CLASS_NAMES
import matplotlib.pyplot as plt

from my_utils.utils import load_model_cancel_parallel, read_image, load_model


class Detector(object):
    def __init__(
            self,
            img_size: tuple,
            model: torch.nn.Module,
    ):
        self.img_size = img_size
        self.transform = transforms.Resize(img_size, antialias=True)
        self.model = model

    def detect(self, img: torch.Tensor, show_window=False, crop=False):
        img_tensor = torch.unsqueeze(img, dim=0)
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
        scores = torch.squeeze(scores).tolist()
        scores = scores if isinstance(scores, list) else [scores]
        labels = [labels[l] + " %.2f" % scores[l] for l in range(len(labels))]
        labels_rst = []
        bbox_rst_list = []
        for i in range(len(scores)):
            if scores[i] >= 0.3:
                labels_rst.append(labels[i])
                bbox_rst_list.append(bbox[i])
        bbox_rst = torch.zeros(len(bbox_rst_list), bbox.shape[1])
        for idx, each in enumerate(bbox_rst_list):
            bbox_rst[idx] = each
        if show_window:
            result = (img_tensor[0] * 255).type(torch.uint8)
            result = draw_bounding_boxes(result, bbox_rst, labels=labels_rst, width=2)
            # result = draw_bounding_boxes(result, bbox, labels=labels, width=2)
            result = result.numpy()
            result = np.transpose(result, (1, 2, 0))
            plt.imshow(result)
            plt.show()


if __name__ == '__main__':
    model = fasterrcnn_resnet50_fpn(
        pretrained=False,
        progress=True,
        num_classes=3,
        pretrained_backbone=True,
        trainable_backbone_layers=5,
    )
    # model = torch.nn.DataParallel(model)
    model = load_model(
        model,
        r'E:\my_files\programmes\python\detector_output\4-13-resnet-2cls\saved_checkpoints\model_param_149.pkl',
        pickle,
        device=torch.device("cpu")
    )
    detector = Detector(
        img_size=(300, 400),
        model=model,
    )
    img_tensor = read_image(r'C:\Users\13769\Desktop\WIN_20220426_19_44_37_Pro.jpg')
    detector.detect(img_tensor, show_window=True)

