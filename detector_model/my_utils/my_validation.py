#!/usr/bin/env python
# coding: utf-8

# In[30]:


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


def get_Weightedf1score(y_real, y_pred):
    # Format conversion
    # y_real_list = y_real.tolist()
    # y_pred_list = y_pred.tolist()

    # number_1 = y_real_list.count([0, 0, 1])
    # number_2 = y_real_list.count([0, 1, 0])
    # number_3 = y_real_list.count([1, 0, 0])

    y_real_conf = []
    y_pred_conf = []
    for item in range(len(y_pred)):
        y_real_conf.append(y_real[item].index(max(y_real[item])))
        y_pred_conf.append(y_pred[item].index(max(y_pred[item])))

    Weighted_f1score = f1_score(y_real_conf, y_pred_conf, average='weighted')
    return Weighted_f1score

# In[38]:


def validation_classifier(y_real,y_pred,label):
    
    n_classes = len(y_real[0])
    
    get_PR_score(y_real,y_pred,label)
    get_PR_img(y_real,y_pred,n_classes)
    
    get_ROC_img(y_real,y_pred,n_classes)
    get_confusionmatrix(y_real,y_pred,label)


# In[39]:


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


# In[48]:


def get_PR_img(y_real,y_pred,n_classes):
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_real[:, i],y_pred[:, i])
        average_precision[i] = average_precision_score(y_real[:, i],y_pred[:, i])

    precision["macro"], recall["macro"], _ = precision_recall_curve(y_real.ravel(), y_pred.ravel())
    average_precision["macro"] = average_precision_score(y_real, y_pred,average="macro")
    
    plt.figure()
    plt.step(recall['macro'], precision['macro'], where='post')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Average precision score, macro-averaged over all classes: AP={0:0.3f}'.format(average_precision["macro"]))


# In[49]:


def get_confusionmatrix(y_real,y_pred,label):
    # Format conversion
    y_real_list = y_real.tolist()
    y_pred_list = y_pred.tolist()

    number_1 = y_real_list.count([0,0,1])
    number_2 = y_real_list.count([0,1,0])
    number_3 = y_real_list.count([1,0,0])

    y_real_conf = []
    y_pred_conf = []
    for item in range(len(y_pred_list)):
        y_real_conf.append(y_real_list[item].index(max(y_real_list[item])))
        y_pred_conf.append(y_pred_list[item].index(max(y_pred_list[item])))  


    cm = confusion_matrix(y_real_conf, y_pred_conf)
    conf_matrix = pd.DataFrame(cm, index=label, columns=label)

    # plot size setting
    fig, ax = plt.subplots(figsize = (4.5,3.5))
    sns.heatmap(conf_matrix, annot=True, annot_kws={"size": 19}, cmap="Blues")
    plt.ylabel('True label', fontsize=18)
    plt.xlabel('Predicted label', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.savefig('confusion.pdf', bbox_inches='tight')
    plt.show()


# In[50]:


def get_PR_score(y_real,y_pred,label):
    
    # Format conversion
    y_real_list = y_real.tolist()
    y_pred_list = y_pred.tolist()

    number_1 = y_real_list.count([0,0,1])
    number_2 = y_real_list.count([0,1,0])
    number_3 = y_real_list.count([1,0,0])

    y_real_conf = []
    y_pred_conf = []
    for item in range(len(y_pred_list)):
        y_real_conf.append(y_real_list[item].index(max(y_real_list[item])))
        y_pred_conf.append(y_pred_list[item].index(max(y_pred_list[item])))  
        
    print(classification_report(y_real_conf, y_pred_conf, target_names=label, digits=4, labels=list(range(len(label)))))


# In[55]:


if __name__ == "__main__":
    pass
    '''
    val_dataset = MultiClassDataset(
        dataset_dir=val_dataset_dir,
        size=img_size,
        classes=class_list,
        resize_mode=resize_mode,
        use_random_augmentation=True,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        shuffle=shuffle,
        num_workers=num_workers,
        batch_size=batch_size,
        # prefetch_factor=4,
    )
    '''
    
#     y_real = torch.tensor([[0,1,0],
#         [0,0,1],
#         [0,0,1],
#         [1,0,0],
#         [0,0,1],
#         [1,0,0],
#         [0,0,1],
#         [0,0,1],
#         [1,0,0],
#         [0,0,1],
#         [1,0,0],
#         [0,0,1],
#         [0,0,1],
#         [0,1,0],
#         [0,1,0],
#         [0,1,0],
#         [0,1,0],
#         [0,0,1],
#         [0,0,1]])
#     y_pred = torch.tensor([[0.33,0.34,0.33],
#         [0.1,0.2,0.7],
#         [0.1,0.3,0.6],
#         [0.4,0.3,0.3],
#         [0.1,0.5,0.4],
#         [0.3,0.5,0.2],
#         [0.1,0.4,0.5],
#         [0.1,0.5,0.4],
#         [0.3,0.5,0.2],
#         [0.3,0.5,0.2],
#         [0.1,0.4,0.5],
#         [0.1,0.5,0.4],
#         [0.3,0.5,0.2],
#         [0.3,0.5,0.2],
#         [0.3,0.5,0.2],
#         [0.1,0.4,0.5],
#         [0.1,0.5,0.4],
#         [0.3,0.5,0.2],
#         [0.1,0.3,0.6]])
    
#     y_real = torch.tensor([[0,1,0,0],
#         [0,0,0,1],
#         [0,0,0,1],
#         [0,1,0,0],
#         [0,0,0,1],
#         [1,0,0,0],
#         [0,0,1,0]])
#     y_pred = torch.tensor([[0.33,0.34,0.11,0.22],
#         [0.1,0.2,0.2,0.5],
#         [0.1,0.3,0.2,0.4],
#         [0.4,0.3,0.1,0.2],
#         [0.1,0.5,0.2,0.2],
#         [0.3,0.5,0.1,0.1],
#         [0.1,0.3,0.1,0.5]])  

#     y_real = torch.tensor([[0,1],
#             [0,1],
#             [0,1],
#             [0,1],
#             [0,1],
#             [1,0],
#             [1,0]])
#     y_pred = torch.tensor([[0.33,0.67],
#         [0.1,0.9],
#         [0.1,0.9],
#         [0.4,0.6],
#         [0.1,0.9],
#         [0.3,0.7],
#         [0.9,0.1]])  

#     y_real = torch.tensor([[0,1,0,0,0],
#         [0,0,0,0,1],
#         [0,0,0,1,0],
#         [0,0,1,0,0],
#         [0,0,0,0,1],
#         [1,0,0,0,0],
#         [0,0,1,0,0]])
#     y_pred = torch.tensor([[0.33,0.34,0.11,0.11,0.11],
#         [0.1,0.2,0.2,0.3,0.2],
#         [0.1,0.3,0.2,0.2,0.2],
#         [0.4,0.3,0.1,0.1,0.1],
#         [0.1,0.5,0.2,0.1,0.1],
#         [0.3,0.5,0.1,0.05,0.05],
#         [0.1,0.3,0.1,0.2,0.3]]) 
        
#     label = ['1','2','3','4','5']
#     validation_classifier(y_real,y_pred,label)
    
    


# In[ ]:




