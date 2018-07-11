#########################################
#
# Helper utilities
# 
#
#########################################
import re, sys, math
import glob
import random
import json
import os 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from collections import OrderedDict





def plot_learning_history(file_p):
    """Function to plot learning history captured during model training.       

    Args:
        file_p(:string): learning history file (.json) with full path.
    
    Returns:
       None.
       
    """
    try:
        with open(file_p, 'r') as file:
            history = json.load(file)
    except (OSError, ValueError):  # file does not exist or is empty/invalid
        print ("File does not exist : ", file_p)
        return
        
    # list all data in history
    #print(history.keys())
    print('-'*30)
    print ("Model Parameters")
    for key in history:
        if (key.startswith("tr_") == True):
            print (key, " : ", history[key])
    print('-'*30)
    print ("Evaluation on Test set")
    for key in history:
        if (key.startswith("eval_") == True):
            print (key, " : ", history[key])
            
    print('-'*30)
    print ("Values at first and last epoch")
    print('-'*30)
    for key in history:
        if (key.startswith("eval_") != True) and (key.startswith("tr_") != True):
            print (key, " : ", history[key][0], ",", history[key][-1])
    print('-'*30) 
    print('-'*30)
    # summarize history for accuracy
    if 'dice_coeff' in history.keys():
        plt.plot(history['dice_coeff'])
        plt.plot(history['val_dice_coeff'])
        plt.title('model accuracy(dice_coeff)')
    elif 'val_acc' in history.keys():
        plt.plot(history['acc'])
        plt.plot(history['val_acc'])
        plt.title('Model accuracy')
    elif 'categorical_accuracy' in history.keys():
        plt.plot(history['categorical_accuracy'])
        plt.plot(history['val_categorical_accuracy'])
        plt.title('categorical_accuracy')
    elif 'binary_accuracy' in history.keys():
        plt.plot(history['binary_accuracy'])
        plt.plot(history['val_binary_accuracy'])
        plt.title('Minary_accuracy')
    else : 
        print ("new loss function, not in the list")
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'val'], loc='lower right')
    plt.grid()
    plt.show()
    # summarize history for loss
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.grid()
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'val'], loc='upper right')
    plt.show()



def show_performance_statistics (y_true_f, y_pred_f):
    """Function to compute and display performanc statistics using labels and prections.       

    Args:
        y_true_f(:string):  label file (.npy) with full path.
        y_pred_f(:string):  predictions file (.npy) with full path.
        
    Returns:
       None.
    
    Note:
        prection file should have the sigmoid outputs (not the rounded values).
    """ 
       
    y_true = np.load(y_true_f)
    y_pred = np.load(y_pred_f)
    #print (y_true.shape, y_pred.shape)
    
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    weight = 0.8
    sample_weights = np.copy(y_true)
    sample_weights[sample_weights == 1] = 1.
    sample_weights[sample_weights == 0] = .2
    
    
    epsilon = 1e-7
    y_pred[y_pred<=0.] = epsilon
    y_pred[y_pred>=1.] = 1. -epsilon
    
    #print (y_true.shape, y_pred.shape)

    score = log_loss (y_true, y_pred)
    score2 = log_loss (y_true, y_pred, sample_weight = sample_weights)
    acc = math.exp(-score)
    acc2 = math.exp(-score2)
    y_pred = np.round(y_pred)
    print('-'*30)
    print ("Loss and Accuracy")
    print('-'*30)
    
    print ("log_loss : ", score,  "  Accuracy: ", acc)
    print ("weighted log_loss : ", score2,  "  Weighted_accuracy: ", acc2)
    print('-'*30)
    
    print ("Model Performance")
    print('-'*30)
    prec = precision_score(y_true, y_pred, average="binary")
    rec = recall_score(y_true, y_pred, average="binary")
    f1 = f1_score(y_true, y_pred, average="binary")
    print("precision :", prec)
    print("recall :", rec) 
    print("f1 score :", f1)
    
    cm = confusion_matrix(y_true, y_pred)
    #cm.print_stats()
    true_p = cm[1][1]
    false_p = cm[0][1]
    true_n = cm[0][0]
    false_n = cm[1][0]
    print ("")
    print ("true_p = %d, false_p = %d, true_neg = %d, false_neg = %d"%(true_p, false_p, true_n, false_n))
    print ("confuion matrix")
    print (cm)
    print ("")
    plt.matshow(cm)
    plt.title('Confusion matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.colorbar()
    plt.show()
    
    print('-'*30)
    print('-'*30)





def find_outliers(y_true_f, y_pred_f):
    """Function find outliers such as labels with zero contours, using labels and prections.       

    Args:
        y_true_f(:string):  label file (.npy) with full path.
        y_pred_f(:string):  predictions file (.npy) with full path.
        
    Returns:
       None.
    
    Note:
        prection file should have the sigmoid outputs (not the rounded values).
    """

    y_true = np.load(y_true_f)
    y_pred_s = np.load(y_pred_f)
    samples, x, y, z = y_true.shape
    print ("Number of Samples : %d, image size : %d x %d "%(samples, x, y))
    y_pred = np.round(y_pred_s)
    y_true_sum = y_true.sum(axis=(1, 2), keepdims=True).reshape(samples)
    y_pred_sum = y_pred.sum(axis=(1, 2), keepdims=True).reshape(samples)  
    lb0 = (np.where(y_true_sum == 0))
    pd0 = (np.where(y_pred_sum == 0))
    lb0 = list(lb0[0])
    pd0 = list(pd0[0])
    print('-'*30)
    print ("Outliers")
    print('-'*30)
    print ("Sample Index of labels with zero contours", lb0)
    print ("Sample Index of predictions with zero contours", pd0)
    ypr = []
    for idx in pd0:
        ypr.append(y_pred_s[idx,:,:,:].max())
    print ("max-sigmoid values with zero contours", ypr)

    img_d = []
    img_j = []
    for i in range(samples) :
        smooth = 0.001
        y_truex = y_true[i].flatten()
        y_predx = y_pred[i].flatten()
        intersection = np.sum(y_truex * y_predx)
        dice_coefx = (2. * intersection + smooth) / (np.sum(y_truex) + np.sum(y_predx) + smooth)
        jaccard_coefx = float(intersection + smooth) / float(np.sum(y_truex) + np.sum(y_predx)-intersection + smooth)
        dice_coefx = np.around(dice_coefx, decimals=3)
        jaccard_coefx = np.around(jaccard_coefx, decimals=3)
        img_d.append(dice_coefx)
        img_j.append(jaccard_coefx)
    

    
    plt.hist(img_d, bins=[i/20 for i in range(20)])
    plt.grid()
    plt.title('Distribution dice coef')
    plt.xlabel('dice_coef')
    plt.ylabel('Sample count')
    plt.show()
    
    plt.hist(img_j, bins=[i/20 for i in range(20)])
    plt.grid()
    plt.title('Distribution of jaccard coef (IoU)')
    plt.xlabel('jaccard_coef (IoU)')
    plt.ylabel('Sample count')
    plt.show()
    
    
    px0 = [i for i,v in enumerate(img_d) if v ==1.0]
    px1 = [i for i,v in enumerate(img_d) if v > .98]
    px25 = [i for i,v in enumerate(img_d) if v <= .7 and v >.5]
    px50 = [i for i,v in enumerate(img_d) if v < .1]
    px100 = [i for i,v in enumerate(img_d) if v == 0]
    print('-'*30)
    print ("Statistics on missed predictions of contour pixels (white pixels)")
    print('-'*30)
    print ("max, min", min(img_d), max(img_d))
    print ("Sample Index where dice coef = 100%",len(px0), px0)
    print ("Sample Index where dice coef >98%",len(px1), px1)
    print ("Sample Index where dice coef 50%-70%",len(px25), px25)
    print ("Sample Index where dice coef <10%", len(px50),px50)
    print ("Sample Index where dice coef = 0%", len(px100),px100)
    print('-'*30)
    print('-'*30)


def display_images_labels (image_file, label_file,  num_images = 4, random_images = False):
    """Function to display images and labels and overlays.       

    Args:
        image_file(:string):  image file (.npy) with full path.
        label_file(:string):  label file (.npy) with full path.
        num_images (:int, optional) : number of images to be displayed, default is 4.
        random_images (:boolean, optional) : if True pick images randomly, else display first n images, default is False.
        
    Returns:
       None.
    
    Note:
        prection file should have the sigmoid outputs (not the rounded values).
    """
    ts = np.load(image_file)
    tl = np.load(label_file)
    samples, x, y, z = tl.shape

    display_list = []

    if random_images == True:
        display_list = random.sample(range(0, samples), num_images)
    else :
        display_list = [i for i in range (num_images)]

    for i in display_list:
        f, axs = plt.subplots(1,3,figsize=(15,15))
        plt.subplot(131),plt.imshow(ts[i].reshape(x, y))
        plt.title('Image '+str(i)), plt.xticks([]), plt.yticks([])
        plt.subplot(132),plt.imshow(tl[i].reshape(x, y))
        plt.title('Label'), plt.xticks([]), plt.yticks([])
        plt.subplot(133),plt.imshow(ts[i].reshape(x, y)), plt.imshow(tl[i].reshape(x, y), 'binary', interpolation='none', alpha=0.3)
        plt.title('Overlay'), plt.xticks([]), plt.yticks([])
        plt.show()

def display_images_labels_predictions (image_file, label_file, pred_file, num_images = 4, image_list = False, random_images = False):
    """Function to display images,labels, predictions and overlays of labels and predictions.       

    Args:
        image_file(:string):  image file (.npy) with full path.
        label_file(:string):  label file (.npy) with full path.
        pred_file(:string):  prediction file (.npy) with full path.
        image_list (:list, optional) : list images to be displayed, if this field is present then num_images and random_images will be ignored.
        num_images (:int, optional) : number of images to be displayed, default is 4.
        random_images (:boolean, optional) : if True pick images randomly, else display first n images, default is False.
        
    Returns:
       None.
    
    Note:
        prection file should have the sigmoid outputs (not the rounded values).
    """
    ts = np.load(image_file)
    tl = np.load(label_file)
    pred = np.load(pred_file)
    samples, x, y, z = pred.shape
    print ("samples, max, min ", samples, pred.max(), pred.min())
    pred2 = np.round(pred)

    ##Print few images wih actual labels and predictions
    display_list = []
    if image_list == False:
        if random_images == True:
            display_list = random.sample(range(0, samples), num_images)
        else :
            display_list = [i for i in range (num_images)]
    else:
        display_list = image_list

    for i in display_list:
        f, axs = plt.subplots(1,4,figsize=(15,15))
        plt.subplot(141),plt.imshow(ts[i].reshape(x, y))
        plt.title('Image '+str(i)), plt.xticks([]), plt.yticks([])
        plt.subplot(142),plt.imshow(tl[i].reshape(x, y))
        plt.title('Label'), plt.xticks([]), plt.yticks([])
        plt.subplot(143),plt.imshow(pred2[i].reshape(x, y))
        plt.title('Prediction'), plt.xticks([]), plt.yticks([])
        plt.subplot(144),plt.imshow(tl[i].reshape(x, y)), plt.imshow(pred2[i].reshape(x, y), 'binary', interpolation='none', alpha=0.5)
        plt.title('Overlay'), plt.xticks([]), plt.yticks([])
        plt.show()


def display_images_predictions (image_file, pred_file,  num_images=4, image_list=False, random_images=False):
    """Function to display images,predictions and overlays of images and predictions.       

    Args:
        image_file(:string):  image file (.npy) with full path.
        pred_file(:string):  prediction file (.npy) with full path.
        image_list (:list, optional) : list images to be displayed, if this field is present then num_images and random_images will be ignored.
        num_images (:int, optional) : number of images to be displayed, default is 4.
        random_images (:boolean, optional) : if True pick images randomly, else display first n images, default is False.
        
    Returns:
       None.
    
    Note:
        prection file should have the sigmoid outputs (not the rounded values).
    """
    ts = np.load(image_file)
    pred = np.load(pred_file)
    samples, x, y, z = pred.shape
    print ("samples, max, min ", samples, pred.max(), pred.min())
    pred2 = np.round(pred)

    display_list = []
    if image_list == False:
        if random_images == True:
            display_list = random.sample(range(0, samples), num_images)
        else :
            display_list = [i for i in range (num_images)]
    else:
        display_list = image_list

    for i in display_list:
        f, axs = plt.subplots(1,3,figsize=(15,15))
        plt.subplot(131),plt.imshow(ts[i].reshape(x, y))
        plt.title('Image '+str(i)), plt.xticks([]), plt.yticks([])
        plt.subplot(132),plt.imshow(pred2[i].reshape(x, y))
        plt.title('Prediction'), plt.xticks([]), plt.yticks([])
        plt.subplot(133),plt.imshow(ts[i].reshape(x, y)), plt.imshow(pred2[i].reshape(x, y), 'binary', interpolation='none', alpha=0.3)
        plt.title('Overlay'), plt.xticks([]), plt.yticks([])
        plt.show()
        
def display_images_predictions2 (image_array, pred_array,  num_images=4, image_list=False, random_images=False):
    """Function to display images,predictions and overlays of images and predictions.       

    Args:
        image_file(:string):  image file (.npy) with full path.
        pred_file(:string):  prediction file (.npy) with full path.
        image_list (:list, optional) : list images to be displayed, if this field is present then num_images and random_images will be ignored.
        num_images (:int, optional) : number of images to be displayed, default is 4.
        random_images (:boolean, optional) : if True pick images randomly, else display first n images, default is False.
        
    Returns:
       None.
    
    Note:
        prection file should have the sigmoid outputs (not the rounded values).
    """
    ts = image_array
    pred = pred_array
    samples, x, y, z = pred.shape
    print ("samples, max, min ", samples, pred.max(), pred.min())
    pred2 = np.round(pred)

    display_list = []
    if image_list == False:
        if random_images == True:
            display_list = random.sample(range(0, samples), num_images)
        else :
            display_list = [i for i in range (num_images)]
    else:
        display_list = image_list

    for i in display_list:
        f, axs = plt.subplots(1,3,figsize=(15,15))
        plt.subplot(131),plt.imshow(ts[i].reshape(x, y))
        plt.title('Image '+str(i)), plt.xticks([]), plt.yticks([])
        plt.subplot(132),plt.imshow(pred2[i].reshape(x, y))
        plt.title('Prediction'), plt.xticks([]), plt.yticks([])
        plt.subplot(133),plt.imshow(ts[i].reshape(x, y)), plt.imshow(pred2[i].reshape(x, y), 'binary', interpolation='none', alpha=0.3)
        plt.title('Overlay'), plt.xticks([]), plt.yticks([])
        plt.show()
        


       
def  display_images_predictions3(image_array, pred_array1, pred_array2,  num_images=4, image_list=False, random_images=False, overlay = True):
    """Function to display images,predictions and overlays of images and predictions.       

    Args:
        image_file(:string):  image file (.npy) with full path.
        pred_file(:string):  prediction file (.npy) with full path.
        image_list (:list, optional) : list images to be displayed, if this field is present then num_images and random_images will be ignored.
        num_images (:int, optional) : number of images to be displayed, default is 4.
        random_images (:boolean, optional) : if True pick images randomly, else display first n images, default is False.
        
    Returns:
       None.
    
    Note:
        prection file should have the sigmoid outputs (not the rounded values).
    """
    ts = image_array
    pred1 = pred_array1
    pred2 = pred_array2
    samples, x, y, z = ts.shape
    print ("samples, max, min ", samples, pred1.max(), pred1.min())
    pred1r = np.round(pred1)
    pred2r = np.round(pred2)

    display_list = []
    if image_list == False:
        if random_images == True:
            display_list = random.sample(range(0, samples), num_images)
        else :
            display_list = [i for i in range (num_images)]
    else:
        display_list = image_list

    for i in display_list:
        f, axs = plt.subplots(1,3,figsize=(15,15))
        plt.subplot(131),plt.imshow(ts[i].reshape(x, y))
        plt.title('Image '+str(i)), plt.xticks([]), plt.yticks([])
        if overlay == True:
            plt.subplot(132),plt.imshow(ts[i].reshape(x, y)), plt.imshow(pred1r[i].reshape(x, y), 'binary', interpolation='none', alpha=0.3)
        else : 
            plt.subplot(132),plt.imshow(pred1r[i].reshape(x, y))
        plt.title('Pred 1'), plt.xticks([]), plt.yticks([])
        if overlay == True:
            plt.subplot(133),plt.imshow(ts[i].reshape(x, y)), plt.imshow(pred2r[i].reshape(x, y), 'binary', interpolation='none', alpha=0.3)
        else : 
            plt.subplot(133),plt.imshow(pred2r[i].reshape(x, y))
        plt.title('Pred 2'), plt.xticks([]), plt.yticks([])
        plt.show()
        
    
        
        
if __name__ == "__main__":
    file_p = "/masvol/heartsmart/unet_model/data/baseline/sunnybrook_1_3_256_learning_history.json"
    plot_accuracy_and_loss(file_p)
