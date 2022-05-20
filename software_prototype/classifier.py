import torch
import numpy as np
import torchvision.transforms.transforms as transforms

import tqdm
from classes import *

from my_utils import load_model_cancel_parallel, read_image
from classifier_model import ClassifierModel


class Classifier(object):
    def __init__(
            self,
            img_size: tuple,
            model: torch.nn.Module,
            show_window: bool,
            class_list: tuple,
    ):
        self.img_size = img_size
        self.transform = transforms.Resize(img_size, antialias=True)
        self.model = model
        self.show_window = show_window
        self.class_list = class_list

    def classify(self, img: torch.Tensor):
        img_tensor = torch.unsqueeze(img, dim=0)
        img_tensor = self.transform(img_tensor)
        self.model.eval()
        prediction = self.model(img_tensor)
        prediction = torch.squeeze(prediction)
        predicted = np.argmax(prediction.detach().numpy())
        # print(prediction)
        # print(predicted)
        return prediction, predicted


if __name__ == '__main__':
    model = ClassifierModel(net_type='vgg', num_classes=len(MASK_CLASSES))
    # model = torch.nn.DataParallel(model)
    '''
    model = load_model_cancel_parallel(
        model,
        r'E:\my_files\programmes\python\detector_output\3-30-classifier-mask\saved_checkpoints\model_param_79.pkl',
        pickle,
        device=torch.device("cpu")
    )
    '''
    classifier = Classifier(
        img_size=(120, 120),
        model=model,
        show_window=True,
        class_list=MASK_CLASSES
    )
    img_tensor = read_image(r'E:\my_files\programmes\python\dp_dataset\mask\wearing\WIN_20220228_09_55_10_Prodrvr_head0.jpg')

    import time
    total = 10000
    time_start = time.time()
    for i in tqdm.tqdm(range(total)):
        test = torch.rand_like(img_tensor)
        _, _ = classifier.classify(test)
    time_end = time.time()
    time_c = time_end - time_start
    print('FPS', total/time_c)
