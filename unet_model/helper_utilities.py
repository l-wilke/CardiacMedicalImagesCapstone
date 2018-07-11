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
import cv2
from sklearn.metrics import log_loss
from sklearn.metrics import auc, roc_curve, accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from collections import OrderedDict
from helpers_dicom import DicomWrapper as dicomwrapper
import skimage
from skimage import measure,feature
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


perf_keys = ["samples", "logloss", "weighted_logloss","accuracy", "weighted_accuracy", "dice_coef", "precision","recall", \
             "f1_score", "true_positive", "false_positive","true_negative","false_negative", "zero_contour_labels", \
             "zero_contour_pred", "missed_pred_lt_05", "missed_pred_gt_25", "missed_pred_gt_50", "missed_pred_eq_100"]

perf_keys2 = ["samples", "logloss", "weighted_logloss","accuracy", "weighted_accuracy",  \
              "dice_coef", "jaccard_coef", "dice_coef2", "jaccard_coef2","precision","recall", \
             "f1_score", "true_positive", "false_positive","true_negative","false_negative", "zero_contour_labels", \
             "zero_contour_pred", "dice_coef_eq_100", "dice_coef_gt_98", "dice_coef_lt_10", "dice_coef_eq_0"]

perf_keys_ext=["tr_model_name","tr_nGPUs", "tr_loss_fn","tr_dropout", "tr_optimizer","tr_lrrate","tr_batchsize","tr_epoch", \
               "tr_size","tr_contrast_norm","tr_augmentation","tr_augment_count","tr_augment_shift_h", "tr_augment_shift_w", \
               "tr_augment_rotation","tr_augment_zoom","eval_loss","eval_dice_coeff","eval_binary_accuracy","logloss", \
               "accuracy","weighted_logloss", "weighted_accuracy","precision","recall","f1_score", "true_positive", \
             "true_negative", "false_positive", "false_negative"]

perf_keys_ext2=["eval_loss","eval_dice_coeff","eval_binary_accuracy","logloss", \
               "accuracy","precision","recall","f1_score", "true_positive", \
             "true_negative", "false_positive", "false_negative"]

def get_perf_keys():
    return perf_keys

def get_perf_keys2():
    return perf_keys2

def get_perf_keys_ext():
    return perf_keys_ext

def get_perf_keys_ext2():
    return perf_keys_ext2


def load_images_and_labels(data, normalize= True, zscore_normalize = False, contrast_normalize= False, cliplimit=2, tilesize=8 ):
    """Function to load images and labels from .npy file into a 4d numpy array
    before we feed them into U-net model. 

    Note:
    Image files will be normalized based on pixel values by default.       

    Args:
        data (:dict): dictionary with full path to .npy files for images and labels.
        normalize (:boolean, optional): Applies pixel normalization if True. Default value is True.
    
    Returns:
       numpy arrays  of images and labels.
    """
    print('-'*30)
    print('load np arrays of images and labels...')
    print('-'*30)
    imgfile = data["images"]
    labelfile = data["labels"]
    print ("Loading files : ", imgfile, labelfile)
    
    im = np.load(imgfile)
    lb = np.load(labelfile)
    if (contrast_normalize == True):
        # Apply contrast normalization
            #create a CLAHE object (Arguments are optional).
        clahe = cv2.createCLAHE(clipLimit=cliplimit, tileGridSize=(tilesize, tilesize))
        print ("CLAHE clip limit, tile size",cliplimit, tilesize )
        imgs_equalized = np.empty(im.shape)
#         for i in range(imgs.shape[0]):
#             imgs_equalized[i,0] = clahe.apply(np.array(imgs[i,0], dtype = np.uint8))
        for i in range(im.shape[0]):
            imgs_equalized[i,:,:,0] = clahe.apply(np.array(im[i,:,:,0], dtype = np.uint16))
            
        print("shape, max, min, mean of original image set:", im.shape, im.max(), im.min(), im.mean())
        im = imgs_equalized
        
        print("shape, max, min, mean of CLAHE image set:", imgs_equalized.shape, imgs_equalized.max(), imgs_equalized.min(), imgs_equalized.mean())
        
    if (normalize == True):
        images = im.astype('float32')
        labels = lb.astype('float32')
        ##Normalize the pixel values, (between 0..1)
        x_min = images.min(axis=(1, 2), keepdims=True)
        x_max = images.max(axis=(1, 2), keepdims=True)
        images2 = (images - x_min)/(x_max-x_min)
        print("shape, max, min, mean of original image set:", images.shape, images.max(), images.min(), images.mean())
        print("shape, max, min, mean after normalization  :", images2.shape, images2.max(), images2.min(), images2.mean())
        print("shape, max, min, mean of labels :", labels.shape, labels.max(), labels.min(), labels.mean())
    elif (zscore_noramlize == True):
        images = im.astype('float32')
        labels = lb.astype('float32')
        ##Normalize the pixel values, (between 0..1)
        x_mean = images.mean(axis=(1, 2), keepdims=True)
        x_std = images.std(axis=(1, 2), keepdims=True)
        images2 = (images - x_mean)/(x_std)
        print("shape, max, min, mean of original image set:", images.shape, images.max(), images.min(), images.mean())
        print("shape, max, min, mean after zscore normalization  :", images2.shape, images2.max(), images2.min(), images2.mean())
        print("shape, max, min, mean of labels :", labels.shape, labels.max(), labels.min(), labels.mean())
    else :
        images2 = im
        labels = lb
        print("shape, max, min, mean of images :", images2.shape, images2.max(), images2.min(), images2.mean())
        print("shape, max, min, mean of labels :", labels.shape, labels.max(), labels.min(), labels.mean())
    
    return images2, labels

def load_images_and_labels_contrast(data, normalize= True, cliplimit=2, tilesize=8):
    """Function to load images and labels from .npy file into a 4d numpy array
    before we feed them into U-net model. 

    Note:
    Image files will be normalized based on pixel values by default.       

    Args:
        data (:dict): dictionary with full path to .npy files for images and labels.
        normalize (:boolean, optional): Applies pixel normalization if True. Default value is True.
    
    Returns:
       numpy arrays  of images and labels.
    """
    print('-'*30)
    print('load np arrays of images and labels...')
    print('-'*30)
    imgfile = data["images"]
    labelfile = data["labels"]
    print ("Loading files : ", imgfile, labelfile)
    
    im = np.load(imgfile)
    lb = np.load(labelfile)
    if (normalize == True):
    # Apply contrast normalization
            #create a CLAHE object (Arguments are optional).
        clahe = cv2.createCLAHE(clipLimit=cliplimit, tileGridSize=(tilesize, tilesize))
        print ("CLAHE clip limit, tile size",cliplimit, tilesize )
        imgs_equalized = np.empty(im.shape)
#         for i in range(imgs.shape[0]):
#             imgs_equalized[i,0] = clahe.apply(np.array(imgs[i,0], dtype = np.uint8))
        for i in range(im.shape[0]):
            imgs_equalized[i,:,:,0] = clahe.apply(np.array(im[i,:,:,0], dtype = np.uint16))
        #return imgs_equalized
        images = imgs_equalized.astype('float32')
        labels = lb.astype('float32')
        ##Normalize the pixel values, (between 0..1)
        x_min = images.min(axis=(1, 2), keepdims=True)
        x_max = images.max(axis=(1, 2), keepdims=True)
        images2 = (images - x_min)/(x_max-x_min)
        print("shape, max, min, mean of original image set:", images.shape, images.max(), images.min(), images.mean())
        print("shape, max, min, mean of CLAHE image set:", imgs_equalized.shape, imgs_equalized.max(), imgs_equalized.min(), imgs_equalized.mean())
        print("shape, max, min, mean after normalization  :", images2.shape, images2.max(), images2.min(), images2.mean())
        print("shape, max, min, mean of labels :", labels.shape, labels.max(), labels.min(), labels.mean())
    
    else :
        images2 = im
        labels = lb
        print("shape, max, min, mean of images :", images2.shape, images2.max(), images2.min(), images2.mean())
        print("shape, max, min, mean of labels :", labels.shape, labels.max(), labels.min(), labels.mean())
    
    return images2, labels


def load_images_and_labels2(data, normalize= True):
    """Function to load images and labels from .npy file into a 4d numpy array
    before we feed them into U-net model The data is normalized using Z-score normalization technique. 

    Note:
    Image files will be normalized based on pixel values by default.       

    Args:
        data (:dict): dictionary with full path to .npy files for images and labels.
        normalize (:boolean, optional): Applies pixel normalization if True. Default value is True.
    
    Returns:
       numpy arrays  of images and labels.
    """
    print('-'*30)
    print('load np arrays of images and labels...')
    print('-'*30)
    imgfile = data["images"]
    labelfile = data["labels"]
    print ("Loading files : ", imgfile, labelfile)
    
    im = np.load(imgfile)
    lb = np.load(labelfile)
    if (normalize == True):
        images = im.astype('float32')
        labels = lb.astype('float32')
        ##Normalize the pixel values, (between 0..1)
        x_mean = images.mean(axis=(1, 2), keepdims=True)
        x_std = images.std(axis=(1, 2), keepdims=True)
        images2 = (images - x_mean)/(x_std)
        print("shape, max, min, mean of original image set:", images.shape, images.max(), images.min(), images.mean())
        print("shape, max, min, mean after normalization  :", images2.shape, images2.max(), images2.min(), images2.mean())
        print("shape, max, min, mean of labels :", labels.shape, labels.max(), labels.min(), labels.mean())
    
    else :
        images2 = im
        labels = lb
        print("shape, max, min, mean of images :", images2.shape, images2.max(), images2.min(), images2.mean())
        print("shape, max, min, mean of labels :", labels.shape, labels.max(), labels.min(), labels.mean())
    
    return images2, labels

def load_images_and_labels_no_preproc(data):
    print('-'*30)
    print('load np arrays of images and labels...')
    print('-'*30)
    imgfile = data["images"]
    labelfile = data["labels"]
    print ("Loading files : ", imgfile, labelfile)

    images = np.load(imgfile)
    labels = np.load(labelfile)
#     im = np.load(imgfile)
#     lb = np.load(labelfile)
#     images = im.astype('float32')
#     labels = lb.astype('float32')
    
#     ##Normalize the pixel values, (between 0..1)
#     x_min = images.min(axis=(1, 2), keepdims=True)
#     x_max = images.max(axis=(1, 2), keepdims=True)
#     images2 = (images - x_min)/(x_max-x_min)

    print("shape, max, min, mean of original image set:", images.shape, images.max(), images.min(), images.mean())
#    print("shape, max, min, mean after normalization  :", images2.shape, images2.max(), images2.min(), images2.mean())
    print("shape, max, min, mean of labels :", labels.shape, labels.max(), labels.min(), labels.mean())
    return images, labels


def load_images(imgfile, normalize= True, zscore_normalize = False, contrast_normalize= False, cliplimit=2, tilesize=8 ):
    """Function to load images  from .npy file into a 4d numpy array
    before we feed them into U-net model. 

    Note:
    Image files will be normalized based on pixel values by default.       

    Args:
        imgfile (:string): .npy file name with full path for images .
        normalize (:boolean, optional): Applies pixel normalization if True. Default value is True.
    
    Returns:
       numpy array  of images.
       
    """
    print('-'*30)
    print('load np arrays of images ...')
    print('-'*30)
    print ("Loading files : ", imgfile)
    
    im = np.load(imgfile)
    if (contrast_normalize == True):
        # Apply contrast normalization
            #create a CLAHE object (Arguments are optional).
        clahe = cv2.createCLAHE(clipLimit=cliplimit, tileGridSize=(tilesize, tilesize))
        print ("CLAHE clip limit, tile size",cliplimit, tilesize )
        imgs_equalized = np.empty(im.shape)
#         for i in range(imgs.shape[0]):
#             imgs_equalized[i,0] = clahe.apply(np.array(imgs[i,0], dtype = np.uint8))
        for i in range(im.shape[0]):
            imgs_equalized[i,:,:,0] = clahe.apply(np.array(im[i,:,:,0], dtype = np.uint16))
            
        print("shape, max, min, mean of original image set:", im.shape, im.max(), im.min(), im.mean())
        im = imgs_equalized
        
        print("shape, max, min, mean of CLAHE image set:", imgs_equalized.shape, imgs_equalized.max(), imgs_equalized.min(), imgs_equalized.mean())
        
    if (normalize == True):
        images = im.astype('float32')
        ##Normalize the pixel values, (between 0..1)
        x_min = images.min(axis=(1, 2), keepdims=True)
        x_max = images.max(axis=(1, 2), keepdims=True)
        images2 = (images - x_min)/(x_max-x_min)
        print("shape, max, min, mean of original image set:", images.shape, images.max(), images.min(), images.mean())
        print("shape, max, min, mean after normalization  :", images2.shape, images2.max(), images2.min(), images2.mean())
    elif (zscore_normalize == True):
        images = im.astype('float32')
        ##Normalize the pixel values, (between 0..1)
        x_mean = images.mean(axis=(1, 2), keepdims=True)
        x_std = images.std(axis=(1, 2), keepdims=True)
        images2 = (images - x_mean)/(x_std)
        print("shape, max, min, mean of original image set:", images.shape, images.max(), images.min(), images.mean())
        print("shape, max, min, mean after zscore normalization  :", images2.shape, images2.max(), images2.min(), images2.mean())
    else :
        images2 = im
        print("shape, max, min, mean of images :", images2.shape, images2.max(), images2.min(), images2.mean())
   
    return images2

def load_images_contrast(imgfile, normalize= True):
    """Function to load images  from .npy file into a 4d numpy array
    before we feed them into U-net model. 

    Note:
    Image files will be normalized based on pixel values by default.       

    Args:
        imgfile (:string): .npy file name with full path for images .
        normalize (:boolean, optional): Applies pixel normalization if True. Default value is True.
    
    Returns:
       numpy array  of images.
       
    """
    print('-'*30)
    print('load np arrays of images ...')
    print('-'*30)
    print ("Loading files : ", imgfile)
    
    im = np.load(imgfile)
    if (normalize == True):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        imgs_equalized = np.empty(imgs.shape)
#         for i in range(imgs.shape[0]):
#             imgs_equalized[i,0] = clahe.apply(np.array(imgs[i,0], dtype = np.uint8))
        for i in range(im.shape[0]):
            imgs_equalized[i,:,:,0] = clahe.apply(np.array(im[i,:,:,0], dtype = np.uint16))
        #return imgs_equalized
        images = imgs_equalized.astype('float32')
        ##Normalize the pixel values, (between 0..1)
        x_min = images.min(axis=(1, 2), keepdims=True)
        x_max = images.max(axis=(1, 2), keepdims=True)
        images2 = (images - x_min)/(x_max-x_min)
        print("shape, max, min, mean of original image set:", images.shape, images.max(), images.min(), images.mean())
        print("shape, max, min, mean of CLAHE image set:", imgs_equalized.shape, imgs_equalized.max(), imgs_equalized.min(), imgs_equalized.mean())
        print("shape, max, min, mean after normalization:", images2.shape, images2.max(), images2.min(), images2.mean())
    else :
        images2 = im
        print("shape, max, min, mean of images:", images2.shape, images2.max(), images2.min(), images2.mean())
   
    return images2

def load_images2(imgfile, normalize= True):
    """Function to load images  from .npy file into a 4d numpy array
    before we feed them into U-net model. 

    Note:
    Image files will be normalized based on pixel values by default.       

    Args:
        imgfile (:string): .npy file name with full path for images .
        normalize (:boolean, optional): Applies pixel normalization if True. Default value is True.
    
    Returns:
       numpy array  of images.
       
    """
    print('-'*30)
    print('load np arrays of images ...')
    print('-'*30)
    print ("Loading files : ", imgfile)
    
    im = np.load(imgfile)
    if (normalize == True):
        images = im.astype('float32')
        ##Normalize the pixel values, (between 0..1)
        x_mean = images.mean(axis=(1, 2), keepdims=True)
        x_std = images.std(axis=(1, 2), keepdims=True)
        images2 = (images - x_mean)/(x_std)
        print("shape, max, min, mean of original image set:", images.shape, images.max(), images.min(), images.mean())
        print("shape, max, min, mean after normalization:", images2.shape, images2.max(), images2.min(), images2.mean())
    else :
        images2 = im
        print("shape, max, min, mean of images:", images2.shape, images2.max(), images2.min(), images2.mean())
   
    return images2

def read_performance_statistics(file_p):
    """Function to read performance statistics captured during model training.       

    Args:
        file_p(:string): performance statistics file (.json) with full path.

    Returns:
       None.
       
    """
    perf_list = ['logloss', 'weighted_logloss', 'accuracy', 'weighted_accuracy','true_positive', 'false_positive', 'true_negative','false_negative', \
                 'precision','recall', 'f1_score' ]
    #perf = OrderedDict.fromkeys(perf_keys_ext)
    try:
        with open(file_p, 'r') as file:
            perf = json.load(file)
    except (OSError, ValueError):  # file does not exist or is empty/invalid
        print ("File does not exist")
        perf = {}
        
    print('-'*30)
    print ("Model Parameters")
    for key in perf:
        if (key.startswith("tr_") == True):
            print (key, " : ", perf[key])
    print('-'*30)
    print ("Evaluation on Test set")
    for key in perf:
        if (key.startswith("eval_") == True):
            print (key, " : ", perf[key])

    print('-'*30)
    for key in perf_list:
        if key in perf.keys():
            print (key, " : ", perf[key])
    print('-'*30) 
    print('-'*30)
    return perf
    # list all data in history
    
def get_performance_statistics(y_true_f, y_pred_f):
    """Function to plot learning history captured during model training.       

    Args:
        file_p(:string): learning history file (.json) with full path.

    Returns:
       perf(:dict): dictionary of perf statistics
       
    """   
    
#     y_true = np.load(y_true_f)
#     y_pred = np.load(y_pred_f)

    y_true = y_true_f.flatten()
    y_pred = y_pred_f.flatten()

    sample_weights = np.copy(y_true)
    sample_weights[sample_weights == 1] = 1.
    sample_weights[sample_weights == 0] = .2
    
    epsilon = 1e-7
    y_pred[y_pred<=0.] = epsilon
    y_pred[y_pred>=1.] = 1. -epsilon
    
    perf = {}
    
    score = log_loss (y_true, y_pred)
    score2 = log_loss (y_true, y_pred, sample_weight = sample_weights)
    perf["logloss"] = score
    perf["weighted_logloss"] = score2
    perf["accuracy"] = math.exp(-score)
    perf["weighted_accuracy"] = math.exp(-score2)

    smooth = 1.
    intersection = np.sum(y_true * y_pred)
    dice_coef = (2. * intersection + smooth) / (np.sum(y_true) + np.sum(y_pred) + smooth)
    perf["dice_coef"] = dice_coef
    
    y_pred = np.round(y_pred)
    perf["precision"] = precision_score(y_true, y_pred, average="binary")
    perf["recall"] = recall_score(y_true, y_pred, average="binary")
    perf["f1_score"] = f1_score(y_true, y_pred, average="binary")

    cm = confusion_matrix(y_true, y_pred)
    perf["true_positive"] = int(cm[1][1])
    perf["false_positive"] = int(cm[0][1])
    perf["true_negative"] = int(cm[0][0])
    perf["false_negative"] = int(cm[1][0])
    
    #cm.print_stats()
    return perf

def compute_roc_auc(y_true_f, y_pred_f):
    """Function to plot learning history captured during model training.       

    Args:
        file_p(:string): learning history file (.json) with full path.

    Returns:
       perf(:dict): dictionary of perf statistics
       
    """   
    
    y_true_a = np.load(y_true_f)
    y_pred_a = np.load(y_pred_f)

    y_true = y_true_a.flatten()
    y_pred = y_pred_a.flatten()
    
    fpr, tpr, threshold = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, threshold, roc_auc
    


    
def compute_performance_statistics (y_true_f, y_pred_f):
    """Function to compute performanc statistics using labels and prections.       

    Args:
        y_true_f(:string):  label file (.npy) with full path.
        y_pred_f(:string):  predictions file (.npy) with full path.
        
    Returns:
       perf(:dict): dictionary of perf statistics.
    
    Note:
        prection file should have the sigmoid outputs (not the rounded values).
    """
    
    y_true = np.load(y_true_f)
    y_pred = np.load(y_pred_f)
    
    y_true_o = np.load(y_true_f)
    y_pred_o = np.load(y_pred_f)
    #print (y_true.shape, y_pred.shape)
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    sample_weights = np.copy(y_true)
    sample_weights[sample_weights == 1] = 1.
    sample_weights[sample_weights == 0] = .2
    
    
    epsilon = 1e-7
    y_pred[y_pred<=0.] = epsilon
    y_pred[y_pred>=1.] = 1. -epsilon
    
    #print (y_true.shape, y_pred.shape)
    smooth = 1.
    intersection = np.sum(y_true * y_pred)
    dice_coef = (2. * intersection + smooth) / (np.sum(y_true) + np.sum(y_pred) + smooth)
    jaccard_coef = float(intersection + smooth) / float(np.sum(y_true) + np.sum(y_pred)-intersection + smooth)
    
    score = log_loss (y_true, y_pred)
    score2 = log_loss (y_true, y_pred, sample_weight = sample_weights)
    acc = math.exp(-score)
    acc2 = math.exp(-score2)
    y_pred = np.round(y_pred)

    prec = precision_score(y_true, y_pred, average="binary")
    rec = recall_score(y_true, y_pred, average="binary")
    f1 = f1_score(y_true, y_pred, average="binary")

    
    cm = confusion_matrix(y_true, y_pred)
    #cm.print_stats()
    true_p = cm[1][1]
    false_p = cm[0][1]
    true_n = cm[0][0]
    false_n = cm[1][0]

    
    #perf = {}
    
#     keys = ["samples", "logloss", "weighted_logloss","accuracy", "weighted_accuracy", "dice_coef", "precision","recall", "f1_score", "true_positive", \
#            "false_positive","true_negative","false_negative", "zero_contour_labels", "zero_contour_pred", \
#            "missed_pred_lt_05", "missed_pred_gt_25", "missed_pred_gt_50", "missed_pred_eq_100"]
    perf = OrderedDict.fromkeys(perf_keys2)
    
    perf["logloss"] = score
    perf["weighted_logloss"] = score2
    perf["accuracy"] = acc
    perf["weighted_accuracy"] = acc2

    perf["dice_coef"] = dice_coef
    perf["jaccard_coef"] = jaccard_coef
    perf["precision"] = prec
    perf["recall"] = rec
    perf["f1_score"] = f1
    perf["true_positive"] = int(cm[1][1])
    perf["false_positive"] = int(cm[0][1])
    perf["true_negative"] = int(cm[0][0])
    perf["false_negative"] = int(cm[1][0])
    
    
    y_true = y_true_o.flatten()
    y_pred = np.round(y_pred_o)
    y_pred = y_pred.flatten()
    
    intersection = np.sum(y_true * y_pred)
    dice_coef2 = (2. * intersection + smooth) / (np.sum(y_true) + np.sum(y_pred) + smooth)
    jaccard_coef2 = float(intersection + smooth) / float(np.sum(y_true) + np.sum(y_pred)-intersection + smooth)
    perf["dice_coef2"] = dice_coef2
    perf["jaccard_coef2"] = jaccard_coef2
    
    y_true = y_true_o
    y_pred = np.round(y_pred_o)
    samples, x, y, z = y_pred.shape
    y_true_sum = y_true.sum(axis=(1, 2), keepdims=True).reshape(samples)
    y_pred_sum = y_pred.sum(axis=(1, 2), keepdims=True).reshape(samples)  
    lb0 = (np.where(y_true_sum == 0))
    pd0 = (np.where(y_pred_sum == 0))
    lb0 = list(lb0[0])
    pd0 = list(pd0[0])
    perf["samples"] = samples
    perf["zero_contour_labels"] = len(lb0)
    perf["zero_contour_pred"] = len(pd0)
    
    img_d = []
    img_j = []
    y_predr = np.round(y_pred_o)
    for i in range(samples) :
        smooth = .01
        y_truex = y_true_o[i].flatten()
        y_predx = y_predr[i].flatten()
        intersection = np.sum(y_truex * y_predx)
        dice_coefx = (2. * intersection + smooth) / (np.sum(y_truex) + np.sum(y_predx) + smooth)
        jaccard_coefx = float(intersection + smooth) / float(np.sum(y_truex) + np.sum(y_predx)-intersection + smooth)
        dice_coefx = np.around(dice_coefx, decimals=3)
        jaccard_coefx = np.around(jaccard_coefx, decimals=3)
        img_d.append(dice_coefx)
        img_j.append(jaccard_coefx)
    px100 = [i for i,v in enumerate(img_d) if v ==1.0]
    px98 = [i for i,v in enumerate(img_d) if v > .98]
    px10 = [i for i,v in enumerate(img_d) if v < .1]
    px0 = [i for i,v in enumerate(img_d) if v == 0]
    perf["dice_coef_eq_100"] = len(px100)
    perf["dice_coef_gt_98"] = len(px98)
    perf["dice_coef_lt_10"] = len(px10)
    perf["dice_coef_eq_0"] = len(px0)
    return perf




    
def find_outliers_in_prediction(y_pred_f):
    """Function to find outliers such as labels with zero contours.       

    Args:
        y_pred_f(:string):  predictions file (.npy) with full path.
        
    Returns:
       None.
    
    Note:
        prection file should have the sigmoid outputs (not the rounded values)
    """
    y_pred_s = np.load(y_pred_f)
    samples, x, y, z = y_pred_s.shape
    print ("Number of Predictions : %d, image size : %d x %d "%(samples, x, y))
    y_pred = np.round(y_pred_s)
    y_pred_sum = y_pred.sum(axis=(1, 2), keepdims=True).reshape(samples)  
    pd0 = (np.where(y_pred_sum == 0))
    pd0 = list(pd0[0])
    print ("Sample Index of predictions with zero contours", pd0)
    ypr = []
    for idx in pd0:
        ypr.append(y_pred_s[idx,:,:,:].max())
    print ("max-sigmoid values with zero contours", ypr)
    print('-'*30)
    
    pd1 = (np.where(y_pred_sum <= 5))
    pd1 = list(pd1[0])
    print ("Sample Index with contour pixels <= 5", pd1)


def dump_and_sort(image_one_file, origpath, newpath):
    """ Get the minimum and the maximum contours from each slice
    Args:
      image_one_file: predicted ones count in json format
      origpath: original dcm numpy array file path
      newpath: output path of the predicted result file
        
    Returns:
       None.
    
    """
    count = 0
    new_dir = os.path.dirname(newpath)

    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

    with open(image_one_file, 'r') as inputs:
        jin = json.load(inputs)
        slicedict = dict()

        for i in sorted(jin):
            count += 1
            rootnode = i.split("/")
            tmp=rootnode[-1].split('_')
            sax = tmp[0] +'_'+ tmp[1]
            frame = tmp[-1]

            if sax in slicedict:
                slicedict[sax].update({frame: jin[i]})
            else:
                slicedict.update({sax: {frame: jin[i]}})

        min_max = get_min_max_slice(slicedict, origpath)
        
        with open(newpath, 'w') as output:
            output.write("{0}\n".format(json.dumps(min_max)))

def get_min_max_slice(slicedict, origpath):
    """
    Figure out the min and max of each slice

    Args:
      slicedict:  slice info
      origpath: original dcm numpy array file path

    Returns: Identified min max info in dict

    """
    identified = {}

    for i in slicedict: #i is sax
        zmin = 9999999
        zmax = 0
        zminframe = ''
        zmaxframe = ''
        zcounts = {}

        for j in slicedict[i]: #j is frame
            zcount = slicedict[i][j]['ones']

            if zcount in zcounts:
                if 'frame' in zcounts[zcount]:
                    zcounts[zcount]['frame'].append(j)
                else:
                    zcounts[zcount].update({'frame':[j]})
            else:
                zcounts.update({zcount: {'frame':[j]}})
            
            if zcount < zmin:
                zmin = zcount
                zminframe = j

            if zcount > zmax:
                zmax = zcount
                zmaxframe = j

        maxpath = i+'/'+zmaxframe.strip('.npy')
        minpath = i+'/'+zminframe.strip('.npy')
        maxsl = None
        minsl = None

        try:
            maxdw = dicomwrapper(origpath, maxpath)
            maxsl = maxdw.slice_location()
            maxst = maxdw.slice_thickness()
        except:
            print ('error max',origpath, maxpath)
            maxsl = None
            maxst = None

        try:
            mindw = dicomwrapper(origpath, minpath)
            minsl = mindw.slice_location()
            minst = mindw.slice_thickness()
        except:
            print ('error min',origpath, minpath)
            minsl = None
            minst = None

        identified[i] = {'zmin':zmin,
                         'zminframe': zminframe,
                         'minSL': minsl,
                         'minST': minst,
                         'zmax': zmax,
                         'zmaxframe': zmaxframe,
                         'maxSL': maxsl,
                         'maxST': maxst,
                        'zcounts': zcounts}

    return identified

 
def combine_acdc_sunnybrook_data(unet_train_dir, acdc_source, sb_source, image_size):
    """
    Combining ACDC and sunnybrook data

    Args:
      slicedict:  slice info
      origpath: original dcm numpy array file path

    Returns: Identified min max info in dict

    """

    train_images = "_1_3_{0}_train_images.npy".format(image_size)
    train_labels = "_1_3_{0}_train_labels.npy".format(image_size)

    test_images = "_1_3_{0}_test_images.npy".format(image_size)
    test_labels = "_1_3_{0}_test_labels.npy".format(image_size)

    acdc_train_data = {}
    acdc_train_data["images"] = unet_train_dir + acdc_source + train_images
    acdc_train_data["labels"] = unet_train_dir + acdc_source + train_labels

    acdc_train_img, acdc_train_lbl = load_images_and_labels_no_preproc(acdc_train_data)

    acdc_test_data = {}
    acdc_test_data["images"] = unet_train_dir + acdc_source + test_images
    acdc_test_data["labels"] = unet_train_dir + acdc_source + test_labels

    acdc_test_img, acdc_test_lbl = load_images_and_labels_no_preproc(acdc_test_data)

    sb_train_data = {}
    sb_train_data["images"] = unet_train_dir + sb_source + train_images
    sb_train_data["labels"] = unet_train_dir + sb_source + train_labels
    sb_train_img, sb_train_lbl = load_images_and_labels_no_preproc(sb_train_data)

    sb_test_data = {}
    sb_test_data["images"] = unet_train_dir + sb_source + test_images
    sb_test_data["labels"] = unet_train_dir + sb_source + test_labels
    sb_test_img, sb_test_lbl = load_images_and_labels_no_preproc(sb_test_data)

    combined_train_img = np.concatenate((acdc_train_img, sb_train_img), axis=0)
    combined_train_lbl = np.concatenate((acdc_train_lbl, sb_train_lbl), axis=0)

    combined_test_img = np.concatenate((acdc_test_img, sb_test_img), axis=0)
    combined_test_lbl = np.concatenate((acdc_test_lbl, sb_test_lbl), axis=0)
    print (combined_train_img.shape, combined_train_lbl.shape,combined_test_img.shape,combined_test_lbl.shape)
    print ("Saving combined files.......")

    tr_img_file = unet_train_dir + "combined_{0}".format(train_images)
    tr_lbl_file = unet_train_dir + "combined_{0}".format(train_labels)
    tst_img_file = unet_train_dir + "combined_{0}".format(test_images)
    tst_lbl_file = unet_train_dir + "combined_{0}".format(test_labels)

    np.save(tr_img_file, combined_train_img)
    np.save(tr_lbl_file, combined_train_lbl)
    np.save(tst_img_file, combined_test_img)
    np.save(tst_lbl_file, combined_test_lbl)
    
def remove_contour(predictions):
    """ 
    At times the model predicts extra more than one contour. Since there is only one LV, we remove           extra predicted contours.
    The pixel density id highest in LV. The numpy array per patient is summed across all the frames.
    The coordinates for max are taken. The contour containing this point is only retained. In case           in any frame the point is not contained in any frame; no contours are kept.
    Args:
      self: The current patient data in processing
    """

    t=predictions
    found_dic={}
    count=0
    #print(t.shape[0])
    x=0
    for i in range(t.shape[0]):
        x=x+t[i,:,:,0]

    print(np.max(x))
    tgt=np.where(x==np.max(x))
    #print('tgt',tgt)
    x1=max(tgt[0])
    y1=max(tgt[1])
    #print('xy',x1,y1)
    point = Point(x1, y1)    

    for t_im1 in predictions:
        found_dic={} 
        t_im1 = t_im1[:, :, 0]        
        dict_shape={}
        cntrs=skimage.measure.find_contours(t_im1,0.1)  
        found = []

        if len(cntrs)>1:
            try:
                for i in range(len(cntrs)):
                    polygon = Polygon(cntrs[i])  
                    if polygon.contains(point):
                        found_dic[i]=1  
                    else:
                        found.append(i)
            except ValueError:            
                print(len(cntrs[i]),cntrs[i])            

        for j in found:
            dict_shape[j]=cntrs[j].shape[0]

            for k in dict_shape:
               x1=math.trunc(np.min(cntrs[k],axis=0)[0])
               y1=math.trunc(np.min(cntrs[k],axis=0)[1])
               x2=math.trunc(np.max(cntrs[k],axis=0)[0])
               y2=math.trunc(np.max(cntrs[k],axis=0)[1])
               t_im1[x1:x2+1, y1:y2+1]=0
    
    return predictions

def get_ones(predictions, image_source_file):
    """ Count up the 1s in predictions """

    print ('l', len(predictions))
    sourcefiles = []
    sourcedict = dict()

    with open(image_source_file, 'r') as sourceinput:
        for i in sourceinput:
            sourcefiles = i.strip().split(',')

    print ('SF',len(sourcefiles))

    for i in sourcefiles:
        sourcedict[i] = {'ones':0} # init, may not have prediction

    for i in range(len(predictions)):
        zcount = np.count_nonzero(predictions[i])
        sourcedict.update({sourcefiles[i]: {'ones':zcount}}) # save ones count for now

    return sourcedict

def remove_ctrsAll(arr):
    """ Another version of contour removal method """
    t=arr
    found_dic={}
    count=0
    #print(t.shape[0])
    x=0
    for i in range(t.shape[0]):
        #print(np.max(t[i,:,:,0]))
        x=x+t[i,:,:,0]
    print(np.max(x))
    tgt=np.where(x==np.max(x))
    #print('tgt',tgt)
    x1=max(tgt[0])
    y1=max(tgt[1])
    print('xy',x1,y1)
    point = Point(x1, y1)    
    print(x[x1,y1])
    icount = 0
    n=0
    for t_im1 in arr:
        found_dic={} 
        t_im1 = t_im1[:, :, 0]        
        icount += 1
        #print('count',count)
        #print(t_im1.shape)
        dict_shape={}
        remove_dict={}
        sorted_keys=[]
        fill_found=[]
        cntrs=skimage.measure.find_contours(t_im1,0.1)  
        for i in range(len(cntrs)):
            #print ('contour:'+str(i)+'  shape[0]:'+str(cntrs[i].shape[0]))
            dict_shape[i]=cntrs[i].shape[0]  
        sorted_keys = sorted(dict_shape, key=dict_shape.get,reverse=True) 

        #print('dict_shape',dict_shape)
        #print('sorted_keys',sorted_keys)
        #if icount==15:
            #break
        found = []
        in_loop=0
        #print('ctrs count',len(cntrs))
        if len(cntrs)>1:
            #print('# of contours identified:::'+str(len(cntrs)), icount)
            #plt.subplot(121),plt.imshow(t_im1,)
            try:

                for i in sorted_keys:
                    #print(type(cntrs[i]),cntrs[i].shape)
                    #print ('cntrs',cntrs[i])
                    in_loop+=1
                    if len(found_dic)==0:
                        polygon = Polygon(cntrs[i])  
                        if polygon.contains(point):
                            found_dic[i]=1
                            for j in range(in_loop,len(sorted_keys)):
                                polygon1 = Polygon(cntrs[sorted_keys[j]]) 
                                if polygon.contains(polygon1):
                                    print('Yes the polygon contains sub polygon')
                                    fill_found.append(sorted_keys[j])
                                else:
                                    found.append(sorted_keys[j])
                            print('Found after polygon contain',found)        
                        #found=list(found_dic.keys())   

                        else:
                            found.append(i)  
                            print('Since polygon did not contain, found is ',found)
                    else:
                        continue 
                        print('since already found a contour, found is', found)
            except ValueError:            
                print(len(cntrs[i]),cntrs[i]) 
            print('found',found)    
        count=count+1    
        
        remove_dict={}

        for k in fill_found:
            print(k)
            print ('filling contour '+str(k))
            x1=math.trunc(np.min(cntrs[k],axis=0)[0])
            y1=math.trunc(np.min(cntrs[k],axis=0)[1])
            x2=math.trunc(np.max(cntrs[k],axis=0)[0])
            y2=math.trunc(np.max(cntrs[k],axis=0)[1])
            t_im1[x1:x2+1, y1:y2+1]=1   
        for k in found:
            print(k)
            print ('deleting contour '+str(k))
            x1=math.trunc(np.min(cntrs[k],axis=0)[0])
            y1=math.trunc(np.min(cntrs[k],axis=0)[1])
            x2=math.trunc(np.max(cntrs[k],axis=0)[0])
            y2=math.trunc(np.max(cntrs[k],axis=0)[1])
            t_im1[x1:x2+1, y1:y2+1]=0
    return arr        

def make_dir(dirpath):
    """ Create the directory of the file path passed in """

    dirname = os.path.dirname(dirpath)
    print ('dirname', dirname)

    if not os.path.exists(dirname):
        os.makedirs(dirname)

def do_ensamble_vote(models, base_dir, out_dir, patient):
    """ Apply ensamble based on the majority vote method """

    print ('models', models)
    model_num = len(models)
    denom = model_num*1.0
    vote_threshold = denom/2 + 1
    image_list=[]
        
    for model in models:
        modelpath = "{0}/{1}_predict/".format(base_dir, model)

        for files in glob.glob(modelpath+'/dsb_'+patient+'*predictions.npy'):
            fpatient = files.split('/')[-1].split('_')[1]
            
            if int(patient) == int(fpatient):                    
                image_list.append(files)
            
    if len(image_list) != int(denom):
        print ('error: not equal')
        return np.array([])
    
    if model_num == 0:
        return np.array([])

    ensamble_pred = None

    for i in range(len(image_list)):
        pred = np.load(image_list[i])
        pred = np.round(pred)

        if i == 0:
            ensamble_pred = pred
        else:
            ensamble_pred = np.add(ensamble_pred, pred)
        
    ensamble_pred = ensamble_pred//vote_threshold
    ensamble_round_pred = "{0}/{1}_ensamble_round_predict.npy".format(out_dir, patient)
    make_dir(ensamble_round_pred)
    np.save(ensamble_round_pred, ensamble_pred)
    return ensamble_pred

def do_ensamble_average(models, base_dir, out_dir, patient):
    """ Apply ensamble based on the average method """

    print ('models', models)
    model_num = len(models)
    denom = model_num*1.0
    image_list=[]
        
    for model in models:
        modelpath = "{0}/{1}_predict/".format(base_dir, model)
        print ('mp', modelpath)

        for files in glob.glob(modelpath+'/dsb_'+patient+'*predictions.npy'):
            fpatient = files.split('/')[-1].split('_')[1]
            
            if int(patient) == int(fpatient):                    
                image_list.append(files)
            
    if len(image_list) != int(denom):
        print ('error: not equal')
        return np.array([])
    
    if model_num == 0:
        return np.array([])

    ensamble_pred = None

    for i in range(len(image_list)):
        if i == 0:
            ensamble_pred = np.load(image_list[i])
        else:
            ensamble_pred = np.add(ensamble_pred, np.load(image_list[i]))
        
    ensamble_predict = "{0}/{1}_ensamble_predict.npy".format(out_dir, patient)
    make_dir(ensamble_predict)
    np.save(ensamble_predict, ensamble_pred)

    ensamble_round = np.round(ensamble_pred)
    ensamble_round_pred = "{0}/{1}_ensamble_round_predict.npy".format(out_dir, patient)
    make_dir(ensamble_round_pred)
    np.save(ensamble_round_pred, ensamble_round)
    return ensamble_round

def post_processing(base_dir, source, input_dir, volume_dir, ensampath, patient, image_size, predictions):
    """ After prediction stage, apply ensamble, contour removal, get ones, and sort out min/max frames in each slice """

    #base_dir = '/masvol/output/dsb/norm/1/3/unet_model_'
    #source = 'train'
    #input_dir = '/masvol/data/dsb'
    #ensampath = '/masvol/output/dsb/norm/1/3/ensamble/top_systolic'
    #                                                 'top_diastolic'
    #volume_dir = '/masvol/output/dsb/volume/1/3path'

    image_source_file = "{0}{1}/data/dsb_{2}_image_path.txt".format(base_dir, source, patient)
    sourcedict = get_ones(predictions, image_source_file)

    pred_file_CR = "{0}/dsb_{1}_{2}_CR4d_predictions_cleaned.npy".format(ensampath, patient, image_size)
    np.save(pred_file_CR, predictions)

    image_one_file = "{0}/dsb_{1}_{2}_one_count.json".format(ensampath, patient, image_size)

    with open(image_one_file, 'w') as output:
        output.write("{0}\n".format(json.dumps(sourcedict)))

    origpath = '{0}/{1}/{2}/study/'.format(input_dir, source, patient)
    newpath = '{0}/{1}_{2}_{3}.json'.format(volume_dir,source,patient,image_size)

    make_dir(newpath)
    dump_and_sort(image_one_file, origpath, newpath)

def do_ensamble(args):
    """ Apply ensamble based on the config file supplied """

    input_dir = args.input_dir
    base_dir = args.base_dir
    sources = args.sources
    output_dir = args.output_dir
    diastolic_models = args.diastolic_models
    systolic_models = args.systolic_models
    systolic_path = args.systolic_path
    diastolic_path = args.diastolic_path
    volume_dir = args.volume_dir
    ensamble_type = args.ensamble_type
    image_size = args.image_size

    for source in sources:
        for patient in glob.glob("{0}/{1}/*".format(input_dir,source)):
            nodes = patient.split('/')
            patient = nodes[-1]

            dia_ensamble_round = None
            sys_ensamble_round = None

            if ensamble_type == 1:
                print ('voting')
                dia_ensamble_round = do_ensamble_vote(diastolic_models, base_dir+source, output_dir+diastolic_path, patient)
                sys_ensamble_round = do_ensamble_vote(systolic_models, base_dir+source, output_dir+systolic_path, patient)
            else:
                dia_ensamble_round = do_ensamble_average(diastolic_models, base_dir+source, output_dir+diastolic_path, patient)
                sys_ensamble_round = do_ensamble_average(systolic_models, base_dir+source, output_dir+systolic_path, patient)

            print ('SYS', sys_ensamble_round.shape, dia_ensamble_round.shape)
            if dia_ensamble_round.shape[0] > 0:
                post_processing(base_dir, source, input_dir, volume_dir, output_dir+diastolic_path, patient, image_size, dia_ensamble_round)

            if sys_ensamble_round.shape[0] > 0:
                post_processing(base_dir, source, input_dir, volume_dir, output_dir+systolic_path, patient, image_size, sys_ensamble_round)


if __name__ == "__main__":
    dummy = 1
