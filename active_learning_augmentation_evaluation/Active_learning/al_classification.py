import torchvision
import torch.nn as nn
import torch.nn.functional
import os
from image_dataset import UnlableDataset
from torch.utils.data import DataLoader
import pickle
from tqdm import tqdm
import pandas as pd
import shutil
import numpy as np
from utils import load_model_cancel_parallel
import pickle

def get_uncertainty_margin_sampling(task_model, unlabeled_loader, labeled_number):
    '''
    margin_sampling(not used in the project)
    '''
    print("Start active learning")
    img_count = 0
    img_length = len(unlabeled_loader)
    score_list = []
    with torch.no_grad():
        
        for imgs in tqdm(unlabeled_loader):             
            img_count = img_count+1
            print("--------------------* Start img ",img_count,"/",img_length," *-----------------------")
            task_model.eval()
            score = task_model(imgs)

            score_list = score.tolist()            
            score_max = 0
            score_min = score_list[0][0]
            for i in range(len(score_list[0])):
                if score_list[0][i] > score_max:
                    score_max = score_list[0][i]
                if score_list[0][i] < score_min:
                    score_min = score_list[0][i]
            score_de = score_max - score_min
            score_list.append(score_de)
        if len(score_list) > labeled_number:
            uncertainty_list = pd.Series(shang_list).sort_values().index[:labeled_number]
        else:
            uncertainty_list = pd.Series(shang_list).sort_values().index[:len(score_list)]
    return uncertainty_list   

def get_uncertainty_entropy(task_model, unlabeled_loader, labeled_number):
    '''
    Entroy method
    '''
    print("Start active learning")
    img_count = 0
    img_length = len(unlabeled_loader)
    shang_list = []
    with torch.no_grad():
        for imgs in tqdm(unlabeled_loader):             
            img_count = img_count+1
            print("--------------------* Start img ",img_count,"/",img_length," *-----------------------")
            task_model.eval()
            score = task_model(imgs)
            score_list = score.tolist() 
            log_probs = np.log2(score_list[0])
            shang = -1 * np.sum(score_list[0] * log_probs, axis=None )
            shang_list.append(shang)
        if len(shang_list) > labeled_number:
            uncertainty_list = pd.Series(shang_list).sort_values().index[:labeled_number]
        else:
            uncertainty_list = pd.Series(shang_list).sort_values().index[:len(shang_list)]
    return uncertainty_list              

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

class ClassifierModel(nn.Module):
    def __init__(self, net_type: str, num_classes: int):
        super(ClassifierModel, self).__init__()
        self.oriNet = None
        self.get_classification_model(net_type, num_classes)

    def get_classification_model(self, net_type, num_classes):
        model = None
        if net_type == "alexNet":
            model = torchvision.models.alexnet(pretrained=True)
            model.classifier[6] = nn.Linear(4096, num_classes)

        elif net_type == "vgg":
            model = torchvision.models.vgg16(pretrained=True)
            model.classifier[6] = nn.Linear(4096, num_classes)

        elif net_type == "googleNet":
            model = torchvision.models.googlenet(pretrained=True)
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)

        elif net_type == "resNet":
            model = torchvision.models.resnet34(pretrained=True)
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)

        elif net_type == "mobileNet":
            model = torchvision.models.mobilenet_v3_large(pretrained=True)
            model.classifier[3] = nn.Linear(1280, num_classes)

        self.oriNet = model

    def forward(self, x: torch.Tensor):
        x = self.oriNet(x)
        outputs = nn.functional.softmax(x, dim=1)
        return outputs

if __name__ == "__main__":
    
    """
        Active learning for classification
        * Aim: Select the most valuable data samples from the unlabeled data set
        * Two methods can be selected:
            Margin sampling: get_uncertainty_margin_sampling
            Entropy: get_uncertainty_entropy
        * Input: 
            dataset_dir: unlabeled dataset
            output_dataset_dir: selected needed labeled dataset
            filename: model store in checkpoints
            labeled_number: Need labeled data sampled number
            ClassifierModel(): Model selected in ResNet, MobileNet, googleNet, vgg, alexNet
         * Output:   
            uncertainty_list: The index number of imge which is in need to labeled
        
    """
    dataset_dir = r'.\data\unlabeled_dataset'
    output_dataset_dir = '.\data\selected_dataset'
    filename = '.\data\model_param_200.pkl'
    labeled_number = 1000
    
    image_name = []
    image_dir = []
    for item in os.listdir(dataset_dir):
        if is_image_file(item):
            image_name.append(item)
            image_dir.append(os.path.join(dataset_dir, item))
    
    unlabeled_dataset = UnlableDataset(
        dataset_dir,
        size=(120, 120),
        resize_mode='stretch',
        use_random_augmentation=True,
    )    
    
    unlabeled_loader = DataLoader(dataset=unlabeled_dataset, batch_size=1,num_workers=0, pin_memory=True)    
        
    model = ClassifierModel('mobileNet', 7)
    pickle_module = pickle
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task_model = load_model_cancel_parallel(model, filename, pickle_module, device)
          
    uncertainty_list = get_uncertainty_entropy(task_model, unlabeled_loader, labeled_number)
    
    for item in uncertainty_list:
        if os.path.exists(output_dataset_dir):
            shutil.copy(str(image_dir[item]), output_dataset_dir)
        else:
            os.makedirs(output_dataset_dir)
            shutil.copy(str(image_dir[item]), output_dataset_dir)

    print("Active learning is over")


