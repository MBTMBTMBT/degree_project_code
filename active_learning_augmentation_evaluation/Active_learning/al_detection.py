from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import pickle
import torch.nn as nn
import torch
import numpy as np
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import cv2 as cv
from torchvision import transforms
import cv2
import os
import shutil
import pandas as pd
import scipy.stats
from utils import load_model_cancel_parallel
from voc_dataset import VOCDataset
from image_dataset import UnlableDataset
from preprocess import _change_light, _change_contrast, _change_hue, _change_saturation, _gaussian_noise, _gaussian_blur

def get_uncertainty(task_model, unlabeled_loader, augs, labeled_num, pickle_dir):
    """
    Choose the picture that has the most information (most needed to be trained)
    :param task_model: Optimal training model
    :param unlabeled_loader: Unlabeled images dataloader from unlabeled pools
    :param augs: Augmentation types
    :param labeled_num: How many images from unlabeled pools want to be selected
    :param pickle_dir: Path of pickle
    """
    
    # Check for checkpoint
    list_file = open(pickle_dir,'rb')
    list2 = pickle.load(list_file)
    if list2[5] == 'finish':
        print("Start active learning")
        img_count = 0
        iou = []
        js = []
        consistency = []
        va_label_rate = []
    elif list2[5] == 'unfinished':
        print("Continue active learning")
        img_count = list2[0]
        iou = list2[1]
        js = list2[2]
        consistency  = list2[3]
        va_label_rate = list2[4]
         
    # Check for augmentation exist
    for aug in augs:
        if aug not in ['is_light', 'is_contrast', 'is_saturation', 'is_hue','is_blur']:
            print('{} is not in the pre-set augmentations!'.format(aug))
           
    img_length = len(unlabeled_loader)    
    task_model.eval()
    dataloader_count = 0
    uncertainty = []
    
    # Start Active learning
    with torch.no_grad():
        for images in unlabeled_loader:
            # Check that the process is correct after the checkpoint
            if dataloader_count < img_count:
                dataloader_count = dataloader_count+1
                continue
            elif dataloader_count > img_count:
                print("dataloader_count error")
            else:
                dataloader_count = dataloader_count+1
                torch.cuda.synchronize()
                aug_images = []
                aug_boxes = []
                for image in images:
                    img_count = img_count+1
                    print("--------------------* Start img ",img_count,"/",img_length," *-----------------------")

                    # * Step1：Make a prediction for the original imgage (Detector D)
                    # Convert image format to prediction
                    img_size = (300, 400)
                    transform = transforms.Resize(img_size, antialias=True)
                    img_tensor = torch.unsqueeze(image, dim=0)
                    img_tensor = transform(img_tensor)
                    # prediction
                    ref_output = task_model(img_tensor)
                    # Get original boxes, scores and lables
                    ref_boxes, ref_labels, ref_scores = ref_output[0]['boxes'], ref_output[0]['labels'], ref_output[0]['scores']
                    # Get Valid and IoU valid boxes, scores and lables (Filter F)
                    va_ref_scores, va_ref_labels, va_ref_boxes,va_IoU_ref_scores, va_IoU_ref_labels, va_IoU_ref_boxes,ref_labels = Filter(ref_scores,ref_labels,ref_boxes) # Get Valid original boxes, scores and lables               

                    # * Step2：Calculate valid information ratio
                    va_label_rate.append(len(va_IoU_ref_labels)/len(va_ref_labels))

                    # * Step3：Augmentation (Image data augmentation A)   
                    if 'is_light' in augs:
                            light_image = _change_light(image)
                            aug_images.append(light_image)
                            aug_boxes.append(va_IoU_ref_boxes)
                    if 'is_contrast' in augs:
                            contrast_image = _change_contrast(image)
                            aug_images.append(contrast_image)
                            aug_boxes.append(va_IoU_ref_boxes)
                    if 'is_saturation' in augs:
                            saturation_image = _change_saturation(image)
                            aug_images.append(saturation_image)
                            aug_boxes.append(va_IoU_ref_boxes)    
                    if 'is_hue' in augs:
                            hue_image = _change_hue(image)
                            aug_images.append(hue_image)
                            aug_boxes.append(va_IoU_ref_boxes)   
                    if 'is_blur' in augs:
                            blur_image = _gaussian_blur(image)
                            aug_images.append(blur_image)
                            aug_boxes.append(va_IoU_ref_boxes)  

                    # * Step4：Make a prediction for the img after augmentation (Detector D)
                    outputs = []               
                    for aug_image in aug_images:
                        aug_img_tensor = torch.unsqueeze(aug_image, dim=0)
                        aug_img_tensor = transform(aug_img_tensor)
                        outputs.append(task_model(aug_img_tensor)[0]) 

                    img_iou = []
                    img_js = []
                    for output, ref_box, aug_image in zip(outputs, aug_boxes, aug_images):
                        aug_iou = []
                        boxes, labels, scores = output['boxes'], output['labels'], output['scores']

                        # Match with reference images (Mather M)
                        va_scores, va_labels, va_boxes, va_IoU_scores, va_IoU_labels, va_IoU_boxes = Matcher(scores,labels,boxes,va_ref_labels,va_ref_boxes,va_IoU_ref_labels,va_IoU_ref_boxes)
                        
                        # Caluculate JS
                        if len(va_IoU_ref_scores) != 0:
                            img_js.append(1-get_JS_main(va_IoU_ref_scores.numpy(),va_IoU_scores.numpy(),5))
                        else:
                            img_js.append(0)
                        
                        for ref_ab,ab,ref_lb,lb,ref_sc,sc in zip(ref_box, va_IoU_boxes, va_IoU_ref_labels, va_IoU_labels, va_IoU_ref_scores, va_IoU_scores):
                            aug_iou.append(get_IoU(ab,ref_ab))
                            
                        # Caluculate IOU
                        tem_iou = 0
                        if len(aug_iou) == 0:
                            continue
                        else: 
                            for item in range(len(aug_iou)):
                                tem_iou = tem_iou + aug_iou[item]
                            img_iou.append(tem_iou/len(aug_iou))

                    tem_iou = 0
                    if len(img_iou) == 0:
                        iou.append(0) 
                    else: 
                        for item in range(len(img_iou)):
                            tem_iou = tem_iou + img_iou[item]
                        iou.append(tem_iou/len(img_iou))
                        
                    tem_js = 0
                    if len(img_js) == 0:
                        js.append(0) 
                    else: 
                        for item in range(len(img_js)):
                            tem_js = tem_js + img_js[item]
                        js.append(tem_js/len(img_js))
                    consistency.append(iou[img_count-1]+js[img_count-1])
                        
                if img_count == img_length:           
                    check_point = [img_count,iou,js,consistency,va_label_rate,'finish']
                else:
                    check_point = [img_count,iou,js,consistency,va_label_rate,'unfinished']
                list_file = open(pickle_dir,'wb')
                pickle.dump(check_point,list_file)
                list_file.close()
        
        # Calculate consistency
        for item in range(len(consistency)):
            uncertainty.append(consistency[item] * va_label_rate[item])
            
        if len(uncertainty)> labeled_num:
            uncertainty_list = pd.Series(uncertainty).sort_values().index[:labeled_num]
        else:
            uncertainty_list = pd.Series(uncertainty).sort_values().index[:len(uncertainty)]
            
    return uncertainty_list

def Filter(ref_scores,ref_labels,ref_boxes):
    """
    Filter the valid image information and get the valid image information rate
    """
    # tensor transform to list
    ref_scores = ref_scores.tolist()
    ref_labels = ref_labels.tolist()
    ref_boxes = ref_boxes.tolist()

    # create result list
    r_ref_scores = []
    r_ref_lables = []
    r_ref_boxes = []
    r_ref_IoU_scores = []
    r_ref_IoU_lables = []
    r_ref_IoU_boxes = []

    ref_scores,ref_labels,ref_boxes = sort(ref_scores,ref_labels,ref_boxes)

    # label = 1: driver_head
    # label = 2: dreiver_hand
    # label = 3: seatbelt

    # obtaine the original image valid detection and valid IoU detection
    for i in set(ref_labels):
        for j in range(len(ref_labels)):
            if ref_labels[j] == i:
                if i == 1:
                    if r_ref_lables.count(i) >= 1:
                        continue
                    else:
                        r_ref_scores.append(ref_scores[j])
                        r_ref_lables.append(ref_labels[j])
                        r_ref_boxes.append(ref_boxes[j])
                        if ref_scores[j] >= 0.6:
                            r_ref_IoU_scores.append(ref_scores[j])
                            r_ref_IoU_lables.append(ref_labels[j])
                            r_ref_IoU_boxes.append(ref_boxes[j])                            
                elif i == 2 or i == 3:
                    if r_ref_lables.count(i) >= 2:
                        continue
                    else:
                        r_ref_scores.append(ref_scores[j])
                        r_ref_lables.append(ref_labels[j])
                        r_ref_boxes.append(ref_boxes[j])
                        if ref_scores[j] >= 0.3:
                            r_ref_IoU_scores.append(ref_scores[j])
                            r_ref_IoU_lables.append(ref_labels[j])
                            r_ref_IoU_boxes.append(ref_boxes[j])
                else:
                    continue
                        
                        
    r_ref_scores = torch.Tensor(r_ref_scores)
    r_ref_lables = torch.Tensor(r_ref_lables)
    r_ref_boxes = torch.Tensor(r_ref_boxes)
    r_ref_IoU_scores = torch.Tensor(r_ref_IoU_scores)
    r_ref_IoU_lables = torch.Tensor(r_ref_IoU_lables)
    r_ref_IoU_boxes = torch.Tensor(r_ref_IoU_boxes)
    
    return r_ref_scores,r_ref_lables,r_ref_boxes,r_ref_IoU_scores,r_ref_IoU_lables,r_ref_IoU_boxes,ref_labels

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


def sort(scores,labels,boxes):
    """
    Sort by image label
    """
    new_scores = []
    new_lables = []
    new_boxes = []

    for i in set(labels):
        for j in range(len(labels)):
            if labels[j] == i:
                new_scores.append(scores[j])
                new_lables.append(labels[j])
                new_boxes.append(boxes[j])
               
    return new_scores,new_lables,new_boxes


def print_img(img,box1,box2,lable1,lable2,score1,score2):
    """
    Image information of reference image and augmentation image is displayed superimposed
    """
    array1=img.numpy()
    array1=array1*255/array1.max()
    mat=np.uint8(array1)
    mat=mat.transpose(1,2,0)
    mat=cv2.cvtColor(mat,cv2.COLOR_BGR2RGB)
    
    left_up_co_1 =(round(400*box1[0].item()),round(300*box1[1].item()))
    right_down_co_1 =(round(400*box1[2].item()),round(300*box1[3].item()))
    right_up_co_1 = (round(400*box1[2].item()),round(300*box1[1].item()))
    left_up_co_2 =(round(400*box2[0].item()),round(300*box2[1].item()))
    right_down_co_2 = (round(400*box2[2].item()),round(300*box2[3].item()))
    right_up_co_2 = (round(400*box2[2].item()),round(300*box2[1].item()))
    
    draw_1 = cv2.rectangle(mat, left_up_co_1, right_down_co_1,(0,0,255), 2)
    draw_2 = cv2.rectangle(draw_1, left_up_co_2, right_down_co_2,(255,0,0), 2)
    
    if lable1 == 1:
        lable_1 = "drvr_head"
    elif lable1 == 2:
        lable_1 = "drvr_hand"
#     elif lable1 == 3:    
#         lable_1 = "seat_belt"
#     elif lable1 == 4:    
#         lable_1 = "mobile_phone"
#     elif lable1 == 5:    
#         lable_1 = "water_or_food"
#     elif lable1 == 6:    
#         lable_1 = "cigarette"
    
    if lable2 == 1:
        lable_2 = "drvr_head"
    elif lable1 == 2:
        lable_2 = "drvr_hand"
#     elif lable1 == 3:    
#         lable_2 = "seat_belt"
#     elif lable1 == 4:    
#         lable_2 = "mobile_phone"
#     elif lable1 == 5:    
#         lable_2 = "water_or_food"
#     elif lable1 == 6:    
#         lable_2 = "cigarette"
      
    font=cv2.FONT_HERSHEY_SIMPLEX 
    draw_lable_1 = cv2.putText(draw_2, '{} {:.3f}'.format(lable_1,score1), right_up_co_1, font, 0.4, (0,0,255), 1)
    draw_lable_2 = cv2.putText(draw_2, '{} {:.3f}'.format(lable_2,score2), right_down_co_2, font, 0.4, (0,0,255), 1)
    
    cv2.imshow('German',draw_2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def Matcher(scores,labels,boxes,va_ref_labels,va_ref_boxes,va_IoU_ref_labels, va_IoU_ref_boxes):
    """
    Augmentation image information was matched one-to-one with reference image information
    """
    scores = scores.tolist()
    labels = labels.tolist()
    boxes = boxes.tolist()
    va_ref_labels = va_ref_labels.tolist()
    va_ref_boxes = va_ref_boxes.tolist()
    va_IoU_ref_labels = va_IoU_ref_labels.tolist()
    va_IoU_ref_boxes = va_IoU_ref_boxes.tolist()

    # create result list
    r_scores = []
    r_lables = []
    r_boxes = []
    r_IoU_scores = []
    r_IoU_lables = []
    r_IoU_boxes = []

    for i in range(len(va_ref_labels)):
        iou = []
        index = -1
        for j in range(len(labels)):
            index += 1
            if labels[j] == va_ref_labels[i]:
                iou.append([get_IoU(boxes[j],va_ref_boxes[i]),index])
        if len(iou) != 0:
            index = iou[iou.index(max(iou))][1]
            r_scores.append(scores[index])
            r_lables.append(labels[index])
            r_boxes.append(boxes[index])
            del scores[index]
            del labels[index]
            del boxes[index]
        else:
            r_scores.append(0)
            r_lables.append(va_ref_labels[i])
            r_boxes.append([0,0,0,0])
    
    tem_r_labels = []
    tem_r_scores = []
    tem_r_boxes = []
        
    for item in range(len(r_lables)):
        tem_r_labels.append(r_lables[item])
        tem_r_scores.append(r_scores[item])
        tem_r_boxes.append(r_boxes[item])
    
    for i in range(len(va_IoU_ref_labels)):
        iou = []
        index = -1
        for j in range(len(tem_r_labels)):
            index += 1
            if tem_r_labels[j] == va_IoU_ref_labels[i]:
                iou.append([get_IoU(tem_r_boxes[j],va_IoU_ref_boxes[i]),index])
        if len(iou) != 0:
            index = iou[iou.index(max(iou))][1]
            r_IoU_scores.append(tem_r_scores[index])
            r_IoU_lables.append(tem_r_labels[index])
            r_IoU_boxes.append(tem_r_boxes[index])
            del tem_r_scores[index]
            del tem_r_labels[index]
            del tem_r_boxes[index]          
        else:
            r_IoU_scores.append(0)
            r_IoU_lables.append(va_IoU_ref_labels[i])
            r_IoU_boxes.append([0,0,0,0])
    
    r_scores = torch.Tensor(r_scores)
    r_lables = torch.Tensor(r_lables)
    r_boxes = torch.Tensor(r_boxes)
    r_IoU_scores = torch.Tensor(r_IoU_scores)
    r_IoU_lables = torch.Tensor(r_IoU_lables)
    r_IoU_boxes = torch.Tensor(r_IoU_boxes)
    
    return r_scores,r_lables,r_boxes ,r_IoU_scores,r_IoU_lables,r_IoU_boxes 

def get_IoU(box,ref_box):
    '''
    Get augmentation images and reference images's IOU
    '''
    
    if torch.is_tensor(box) == False:
        box = torch.tensor(box)
    if torch.is_tensor(ref_box) == False:
        ref_box = torch.tensor(ref_box)

    ref_left_x = round(400*ref_box[0].item())
    ref_left_y = round(300*ref_box[1].item())
    ref_right_x = round(400*ref_box[2].item())
    ref_right_y = round(300*ref_box[3].item())
    left_x = round(400*box[0].item())
    left_y = round(300*box[1].item())
    right_x = round(400*box[2].item())
    right_y = round(300*box[3].item())

    width = min(ref_right_x, right_x) - max(ref_left_x, left_x)
    height = min(ref_right_y, right_y) - max(ref_left_y, left_y)
    ref_area = (ref_right_x - ref_left_x) * (ref_right_y - ref_left_y)
    area = (right_x - left_x) * (right_y - left_y)

    # Iner area is overlap area
    if width <0 or height < 0:
        iner_area = 0
    else: 
        iner_area = width * height

    # Calculate IoU
    iou = iner_area / (ref_area + area - iner_area)      

    return iou

def get_JS_main(arr1,arr2,num_bins):
    """
    Calculate JS
    """
    max0 = max(np.max(arr1),np.max(arr2))
    min0 = min(np.min(arr1),np.min(arr2))
    bins = np.linspace(min0-1e-4, max0+1e-4, num=num_bins)
    PDF1 = pd.cut(arr1,bins).value_counts() / len(arr1)
    PDF2 = pd.cut(arr2,bins).value_counts() / len(arr2)
    return get_JS_score(PDF1.values,PDF2.values)


def get_JS_score(p,q):
    M=(p+q)/2
    return 0.5*scipy.stats.entropy(p, M, base=2)+0.5*scipy.stats.entropy(q, M, base=2)


if __name__ == "__main__":
    
    dataset_dir = r'.\data\unlabeled_dataset'
    output_dataset_dir = r'.\data\selected_dataset'
    image_name = []
    image_dir = []
    for item in os.listdir(dataset_dir):
        if is_image_file(item):
            image_name.append(item)
            image_dir.append(os.path.join(dataset_dir, item))
    
    unlabeled_dataset = UnlableDataset(
        dataset_dir,
        size=(300, 400),
        resize_mode='stretch',
        use_random_augmentation=True,
    )    
    unlabeled_loader = DataLoader(dataset=unlabeled_dataset, batch_size=1,num_workers=0, pin_memory=True)    
        
    model = fasterrcnn_resnet50_fpn(
        pretrained=False,
        progress=True,
        num_classes=7,
        pretrained_backbone=True,
        trainable_backbone_layers=5,
    )
    filename =  r"E:\graduate_project\_active_learning\_objectdetection\model\model_param_1027.pkl"
#     filename = r'.\data\model_param_1027.pkl'
    pickle_module = pickle
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task_model = load_model_cancel_parallel(model, filename, pickle_module, device)
    
    augs = []
    augs.append('is_light')
    # augs.append('is_contrast')
    augs.append('is_hue')
    # augs.append('is_saturation')
    augs.append('is_blur')
    
    labeled_num = 1
    
    pickle_dir = r'.\data\checkpoint.pickle'
    
    uncertainty_list = get_uncertainty(task_model, unlabeled_loader, augs, labeled_num, pickle_dir)
    
    for item in uncertainty_list:
        if os.path.exists(output_dataset_dir):
            shutil.copy(str(image_dir[item]), output_dataset_dir)
        else:
            os.makedirs(output_dataset_dir)
            shutil.copy(str(image_dir[item]), output_dataset_dir)

    print("Active learning is over")


