import cv2
import myfatigue
import emotions
import time
import pickle
import torch
from PySide2 import QtWidgets,QtCore,QtGui
from PySide2.QtWidgets import QMainWindow, QFileDialog, QMessageBox
from PySide2.QtCore import QDir, QTimer,Slot
from PySide2.QtGui import QPixmap,QImage
from ui_mainwindow import Ui_MainWindow
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from my_utils import load_model, load_model_cancel_parallel
from detector import BoundingBoxDetector, MediapipeDetector
from classifier_model import ClassifierModel
from classes import *
from classifier import Classifier
from behaviour_detect import DetectorClassifier


HAS_FACE = False
COORDINATE = 0,0,0,0
EMOTION_MODE = 'neutral'
COLOR = 0
# cap = cv2.VideoCapture(0)

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
hand_classifier_model = ClassifierModel('resNet', len(HEAD_MOTIONS))
hand_classifier_model = load_model_cancel_parallel(
    hand_classifier_model,
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
hand_motion_classifier = Classifier(
    img_size=(120, 120),
    model=hand_classifier_model,
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
detector_classifier = DetectorClassifier(
    detector=detector,
    head_motion_classifier=head_motion_classifier,
    hand_motion_classifier=hand_motion_classifier,
    sunglasses_classifier=sunglasses_classifier,
    head_mask_classifier=head_mask_classifier,
)


def frametest(frame,Roll):
    global HAS_FACE,COORDINATE,EMOTION_MODE,COLOR
    ret = []
    labellist = []

    # timer starts which used to calculate the fps
    tstart = time.time()

    # Counters for blinking and yawning
    if Roll % 5 == 0:
        COORDINATE,EMOTION_MODE,COLOR,HAS_FACE  = emotions.detemotions(frame)
    frame,TOTAL,mTOTAL,COUNTER,mCOUNTER,NO_EYE_COUNTER = myfatigue.detfatigue(frame)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    head_class, sunglasses, mask = detector_classifier.detect_and_classify(frame_rgb)
    # print(head_class, sunglasses, mask)

    if HAS_FACE:
        x, y, width, height = COORDINATE
        cv2.rectangle(frame, (x, y), (x + width, y + height), COLOR, 2)
        cv2.putText(frame, EMOTION_MODE, (x + 0, y - 45),
            cv2.FONT_HERSHEY_SIMPLEX,1, COLOR, 1, cv2.LINE_AA)

    # append those information into ret

    ret.append(int(TOTAL))
    ret.append(int(mTOTAL))
    ret.append(int(COUNTER))
    ret.append(int(mCOUNTER))
    ret.append(int(NO_EYE_COUNTER))

    # timer ends
    tend = time.time()
    # calculate fps
    fps = 1 / (tend-tstart)
    fps_str = "%.2f fps" % fps
    # show fps by cv2
    cv2.putText(frame,fps_str,(10, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)
    
    # return this frame and counters
    return head_class, sunglasses, mask, ret, frame