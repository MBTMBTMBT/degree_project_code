import torch
import numpy
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns
from sklearn import metrics 
from sklearn.metrics import accuracy_score, average_precision_score,precision_score,f1_score,recall_score,roc_curve,auc,precision_recall_curve
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

def validation_classifier(y_real,y_pred,label):
    n_classes = len(y_real[0])
#     Macro_f1score = get_Macrof1score(y_real,y_pred)
#     Weighted_f1score = get_Weightedf1score(y_real,y_pred)
#     accuracy = get_accuracy(y_real,y_pred)
    get_PR_score(y_real,y_pred,label)
#     get_PR_img(y_real,y_pred,n_classes)
#     get_PR_img_sp(y_real,y_pred,n_classes)
#     get_ROC_img(y_real,y_pred,n_classes)
    get_confusionmatrix(y_real,y_pred,label)

def get_PR_score(y_real,y_pred,label):
    # Format conversion
    y_real_list = y_real.tolist()
    y_pred_list = y_pred.tolist()

    y_real_conf = []
    y_pred_conf = []
    for item in range(len(y_pred_list)):
        y_real_conf.append(y_real_list[item].index(max(y_real_list[item])))
        y_pred_conf.append(y_pred_list[item].index(max(y_pred_list[item])))  
        
    print(classification_report(y_real_conf, y_pred_conf, target_names=label, digits=4, labels=list(range(len(label)))))

def get_PR_img(y_real,y_pred,n_classes):
    precision = dict()
    recall = dict()
    average_precision = dict()   
    
    precision["macro"], recall["macro"], _ = precision_recall_curve(y_real.ravel(), y_pred.ravel())
    average_precision["macro"] = average_precision_score(y_real, y_pred,average="macro")
    plt.figure()
    plt.step(recall['macro'], precision['macro'], where='post')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Average precision score, macro-averaged over all classes: AP={0:0.3f}'.format(average_precision["macro"]))

def get_PR_img_sp(y_real,y_pred,n_classes):
    precision = dict()
    recall = dict()
    average_precision = dict()
    
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_real[:, i],y_pred[:, i])
        average_precision[i] = average_precision_score(y_real[:, i],y_pred[:, i])
    
    plt.figure()
    
    for i in range(n_classes):
        if i == 0:
            plt.plot(precision[i], recall[i], lw=2,
                     label='PR curve of class {0} (area = {1:0.2f})'
                     ''.format(i, average_precision[i]))
        elif i == 2:
            plt.plot(precision[i], recall[i], lw=2,
                     label='PR curve of class {0} (area = {1:0.2f})'
                     ''.format(i, average_precision[i]))
        elif i == 3:
            plt.plot(precision[i], recall[i],lw=2,
                     label='PR curve of class {0} (area = {1:0.2f})'
                     ''.format(i, average_precision[i]))
            
    plt.plot([0, 1], [1, 0], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('(ROC) Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()

def get_ROC_img(y_real,y_pred,n_classes):
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_real[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_real.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += numpy.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('(ROC) Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()

def get_confusionmatrix(y_real,y_pred,label):
    # Format conversion
    y_real_list = y_real.tolist()
    y_pred_list = y_pred.tolist()

    y_real_conf = []
    y_pred_conf = []
    for item in range(len(y_pred_list)):
        y_real_conf.append(y_real_list[item].index(max(y_real_list[item])))
        y_pred_conf.append(y_pred_list[item].index(max(y_pred_list[item])))  

    cm = confusion_matrix(y_real_conf, y_pred_conf)
    conf_matrix = pd.DataFrame(cm, index=label, columns=label)

    # plot size setting
    fig, ax = plt.subplots(figsize = (4.5,3.5))
    sns.heatmap(conf_matrix, annot=True, fmt='.20g',annot_kws={"size": 10}, cmap="Blues")
    plt.ylabel('True label', fontsize=12)
    plt.xlabel('Predicted label', fontsize=12)
#     plt.xticks(fontsize=18)
#     plt.yticks(fontsize=18)
    tick_marks = np.arange(len(label))
    plt.xticks(tick_marks, label, fontsize=10)
    plt.yticks(tick_marks, label, fontsize=10)
    plt.savefig('E:\graduate_project\_evaluation\_figure\confusion_matrix.jpg',bbox_inches = 'tight')
    plt.show()

def get_Macrof1score(y_real,y_pred):
    # Format conversion
    y_real_list = y_real.tolist()
    y_pred_list = y_pred.tolist()

    y_real_conf = []
    y_pred_conf = []
    
    for item in range(len(y_pred_list)):
        y_real_conf.append(y_real_list[item].index(max(y_real_list[item])))
        y_pred_conf.append(y_pred_list[item].index(max(y_pred_list[item])))  
    
    Macro_f1score =  f1_score(y_real_conf, y_pred_conf, average='macro')
    return Macro_f1score

def get_Weightedf1score(y_real,y_pred):
    # Format conversion
    y_real_list = y_real.tolist()
    y_pred_list = y_pred.tolist()

    y_real_conf = []
    y_pred_conf = []
    
    for item in range(len(y_pred_list)):
        y_real_conf.append(y_real_list[item].index(max(y_real_list[item])))
        y_pred_conf.append(y_pred_list[item].index(max(y_pred_list[item])))  
    
    weighted_f1score =  f1_score(y_real_conf, y_pred_conf, average='weighted')
    return weighted_f1score

def get_accuracy(y_real,y_pred):
    # Format conversion
    y_real_list = y_real.tolist()
    y_pred_list = y_pred.tolist()

    y_real_conf = []
    y_pred_conf = []
    
    for item in range(len(y_pred_list)):
        y_real_conf.append(y_real_list[item].index(max(y_real_list[item])))
        y_pred_conf.append(y_pred_list[item].index(max(y_pred_list[item])))  
    
    accuracy = accuracy_score(y_real_conf, y_pred_conf, normalize=True, sample_weight=None)
    return accuracy

def PR_model_comparation(y_real_alex,y_pred_alex, y_real_google, y_pred_google, y_real_mobilenet, y_pred_mobilenet, y_real_vgg, y_pred_vgg, y_real_resnet, y_pred_resnet):
    
    n_classes = len(y_real_alex[0])
    
    precision_alex = dict()
    recall_alex = dict()
    average_precision_alex = dict()
    
    precision_google = dict()
    recall_google = dict()
    average_precision_google = dict()
    
    precision_mobilenet = dict()
    recall_mobilenet = dict()
    average_precision_mobilenet = dict()
    
    precision_vgg = dict()
    recall_vgg = dict()
    average_precision_vgg = dict()
    
    precision_resnet = dict()
    recall_resnet = dict()
    average_precision_resnet = dict()
    
    precision_alex["macro"], recall_alex["macro"], _ = precision_recall_curve(y_real_alex.ravel(), y_pred_alex.ravel())
    average_precision_alex["macro"] = average_precision_score(y_real_alex, y_pred_alex,average="macro")
    precision_google["macro"], recall_google["macro"], _ = precision_recall_curve(y_real_google.ravel(), y_pred_google.ravel())
    average_precision_google["macro"] = average_precision_score(y_real_google, y_pred_google,average="macro")
    
    precision_mobilenet["macro"], recall_mobilenet["macro"], _ = precision_recall_curve(y_real_mobilenet.ravel(), y_pred_mobilenet.ravel())
    average_precision_mobilenet["macro"] = average_precision_score(y_real_mobilenet, y_pred_mobilenet,average="macro")
    precision_vgg["macro"], recall_vgg["macro"], _ = precision_recall_curve(y_real_vgg.ravel(), y_pred_vgg.ravel())
    average_precision_vgg["macro"] = average_precision_score(y_real_vgg, y_pred_vgg,average="macro")
    precision_resnet["macro"], recall_resnet["macro"], _ = precision_recall_curve(y_real_resnet.ravel(), y_pred_resnet.ravel())
    average_precision_resnet["macro"] = average_precision_score(y_real_resnet, y_pred_resnet,average="macro")

    plt.figure()
    plt.step(recall_resnet['macro'], precision_resnet['macro'], where='post',label='ResNet')
    plt.step(recall_mobilenet['macro'], precision_mobilenet['macro'], where='post',label='MobileNet')
    plt.step(recall_alex['macro'], precision_alex['macro'], where='post',label='AlexNet')
    plt.step(recall_vgg['macro'], precision_vgg['macro'], where='post',label='VGG16')
    plt.step(recall_google['macro'], precision_google['macro'], where='post',label='GoogLeNet')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])
#     plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.legend(loc="lower left")
    plt.savefig('E:\graduate_project\_evaluation\_figure\PR_model_comparation.jpg')
#     plt.title('Average precision score, macro-averaged over all classes: AP={0:0.3f}'.format(average_precision["macro"]))

def PR_model_comparation_sp(y_real_alex,y_pred_alex, y_real_google, y_pred_google, y_real_mobilenet, y_pred_mobilenet, y_real_vgg, y_pred_vgg, y_real_resnet, y_pred_resnet):
    
    n_classes = len(y_real_alex[0])
    
    precision_alex = dict()
    recall_alex = dict()
    average_precision_alex = dict()
    
    precision_google = dict()
    recall_google = dict()
    average_precision_google = dict()
    
    precision_mobilenet = dict()
    recall_mobilenet = dict()
    average_precision_mobilenet = dict()
    
    precision_vgg = dict()
    recall_vgg = dict()
    average_precision_vgg = dict()
    
    precision_resnet = dict()
    recall_resnet = dict()
    average_precision_resnet = dict()
    
    for i in range(n_classes):
        precision_alex[i], recall_alex[i], _ = precision_recall_curve(y_real_alex[:, i],y_pred_alex[:, i])
        average_precision_alex[i] = average_precision_score(y_real_alex[:, i],y_pred_alex[:, i])    
    for i in range(n_classes):
        precision_google[i], recall_google[i], _ = precision_recall_curve(y_real_google[:, i],y_pred_google[:, i])
        average_precision_google[i] = average_precision_score(y_real_google[:, i],y_pred_google[:, i])
    for i in range(n_classes):
        precision_mobilenet[i], recall_mobilenet[i], _ = precision_recall_curve(y_real_mobilenet[:, i],y_pred_mobilenet[:, i])
        average_precision_mobilenet[i] = average_precision_score(y_real_mobilenet[:, i],y_pred_mobilenet[:, i])
    for i in range(n_classes):
        precision_vgg[i], recall_vgg[i], _ = precision_recall_curve(y_real_vgg[:, i],y_pred_vgg[:, i])
        average_precision_vgg[i] = average_precision_score(y_real_vgg[:, i],y_pred_vgg[:, i])
    for i in range(n_classes):
        precision_resnet[i], recall_resnet[i], _ = precision_recall_curve(y_real_resnet[:, i],y_pred_resnet[:, i])
        average_precision_resnet[i] = average_precision_score(y_real_resnet[:, i],y_pred_resnet[:, i])
    
    plt.figure()
    plt.plot(precision_resnet[2], recall_resnet[2], lw=2,
                     label='ResNet(area = {1:0.2f})'
                     ''.format(i, average_precision_resnet[2]))
    plt.plot(precision_alex[2], recall_alex[2], lw=2,
                     label='AlexNet(area = {1:0.2f})'
                     ''.format(i, average_precision_alex[2]))
    plt.plot(precision_google[2], recall_google[2], lw=2,
                     label='GoogLeNet(area = {1:0.2f})'
                     ''.format(i, average_precision_google[2]))
    plt.plot(precision_mobilenet[2], recall_mobilenet[2], lw=2,
                     label='MobileNet(area = {1:0.2f})'
                     ''.format(i, average_precision_mobilenet[2]))
    plt.plot(precision_vgg[2], recall_vgg[2], lw=2,
                     label='VGG16(area = {1:0.2f})'
                     ''.format(i, average_precision_vgg[2]))

    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.legend(loc="lower right")
    plt.title('P-R curve of no_motion')
    
    
    plt.figure()
    plt.plot(precision_resnet[0], recall_resnet[0], lw=2,
                     label='ResNet(area = {1:0.2f})'
                     ''.format(i, average_precision_resnet[0]))
    plt.plot(precision_alex[0], recall_alex[0], lw=2,
                     label='AlexNet(area = {1:0.2f})'
                     ''.format(i, average_precision_alex[0]))
    plt.plot(precision_google[0], recall_google[0], lw=2,
                     label='GoogLeNet(area = {1:0.2f})'
                     ''.format(i, average_precision_google[0]))
    plt.plot(precision_mobilenet[0], recall_mobilenet[0], lw=2,
                     label='MobileNet(area = {1:0.2f})'
                     ''.format(i, average_precision_mobilenet[0]))
    plt.plot(precision_vgg[0], recall_vgg[0], lw=2,
                     label='VGG16(area = {1:0.2f})'
                     ''.format(i, average_precision_vgg[0]))
    
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.legend(loc="lower right")
    plt.title('P-R curve of eating_or_drinking')
    
    plt.figure()
    plt.plot(precision_resnet[3], recall_resnet[3], lw=2,
                     label='ResNet(area = {1:0.2f})'
                     ''.format(i, average_precision_resnet[3]))
    plt.plot(precision_alex[3], recall_alex[3], lw=2,
                     label='AlexNet(area = {1:0.2f})'
                     ''.format(i, average_precision_alex[3]))
    plt.plot(precision_google[3], recall_google[3], lw=2,
                     label='GoogLeNet(area = {1:0.2f})'
                     ''.format(i, average_precision_google[3]))
    plt.plot(precision_mobilenet[3], recall_mobilenet[3], lw=2,
                     label='MobileNet(area = {1:0.2f})'
                     ''.format(i, average_precision_mobilenet[3]))
    plt.plot(precision_vgg[3], recall_vgg[3], lw=2,
                     label='VGG16(area = {1:0.2f})'
                     ''.format(i, average_precision_vgg[3]))
              
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.legend(loc="lower right")
    plt.title('P-R curve of talking_on_the_phone')

if __name__ == "__main__":
    
    """
        Evaluation of the classification models
        * Input: 
            y_real: True label in tensor (torch.tensor([[0,1,0],[0,0,1])))
            y_pred: predict label in tensor (torch.tensor([[0.1,0.6,0.3],[0.3,0.2,0.5])))
            label: model store in checkpoints (['no_motion', 'looking_away', 'eating_or_drinking'])
         * Output:   
            evaluation result you want
        
    """
    y_real_alex = torch.load('.\_alex\labels.pt')
    y_pred_alex = torch.load('.\_alex\predictions.pt')
    
    y_real_google = torch.load('.\_google\labels.pt')
    y_pred_google = torch.load('.\_google\predictions.pt')
    
    y_real_mobilenet = torch.load('.\_mobilenet\labels.pt')
    y_pred_mobilenet = torch.load('.\_mobilenet\predictions.pt')
    
    y_real_vgg = torch.load('.\_vgg\labels.pt')
    y_pred_vgg = torch.load('.\_vgg\predictions.pt')
    
    y_real_resnet = torch.load('.\_resnet\labels.pt')
    y_pred_resnet = torch.load('.\_resnet\predictions.pt')
    
    label = ['no_motion', 'looking_away', 'eating_or_drinking', 'talking_on_the_phone', 'smoking', 'touch_face_or_head', 'tired']

# Select needed evaluate method
    print("resnet")
    validation_classifier(y_real_resnet,y_pred_resnet,label)
    print("alex")
    validation_classifier(y_real_alex,y_pred_alex,label)
    print("google")
    validation_classifier(y_real_google,y_pred_google,label)
    print("mobilenet")
    validation_classifier(y_real_mobilenet,y_pred_mobilenet,label)
    print("vgg")
    validation_classifier(y_real_vgg,y_pred_vgg,label)
    
#     PR_model_comparation(y_real_alex,y_pred_alex, y_real_google, y_pred_google , y_real_mobilenet, y_pred_mobilenet, y_real_vgg, y_pred_vgg, y_real_resnet, y_pred_resnet)
#     PR_model_comparation_sp(y_real_alex,y_pred_alex, y_real_google, y_pred_google , y_real_mobilenet, y_pred_mobilenet, y_real_vgg, y_pred_vgg, y_real_resnet, y_pred_resnet)




