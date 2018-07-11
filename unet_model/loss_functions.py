
import re, sys, math
import glob
import random
import json
import os 
import numpy as np
from sklearn.metrics import log_loss

from keras.models import *
from keras.layers import Input, concatenate, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.losses import binary_crossentropy
import keras.backend as K
from keras import backend as keras

#########################################
#
# Loss functions
#
#########################################

def dice_coeff(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score


def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss

def log_dice_loss(y_true, y_pred):
    loss =  -K.log(dice_coeff(y_true, y_pred))
    return loss

def dice_loss2(y_true, y_pred):
    loss = -dice_coeff(y_true, y_pred)
    return loss

def bce_dice_loss(y_true, y_pred):
    loss = binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss

def jaccard_distance_loss(y_true, y_pred, smooth=100):
    """
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
    
    The jaccard distance loss is usefull for unbalanced datasets. This has been
    shifted so it converges on 0 and is smoothed to avoid exploding or disapearing
    gradient.
    
    Ref: https://en.wikipedia.org/wiki/Jaccard_index
    
    @url: https://gist.github.com/wassname/f1452b748efcbeb4cb9b1d059dce6f96
    @author: wassname
    """
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth


def weighted_dice_coeff(y_true, y_pred, weight):
    smooth = 1.
    w, m1, m2 = weight * weight, y_true, y_pred
    intersection = (m1 * m2)
    score = (2. * K.sum(w * intersection) + smooth) / (K.sum(w * m1) + K.sum(w * m2) + smooth)
    return score


def weighted_dice_loss(y_true, y_pred):
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    # if we want to get same size of output, kernel size must be odd number
    if K.int_shape(y_pred)[1] == 128:
        kernel_size = 11
    elif K.int_shape(y_pred)[1] == 256:
        kernel_size = 21
    elif K.int_shape(y_pred)[1] == 512:
        kernel_size = 21
    elif K.int_shape(y_pred)[1] == 1024:
        kernel_size = 41
    else:
        raise ValueError('Unexpected image size')
    averaged_mask = K.pool2d(
        y_true, pool_size=(kernel_size, kernel_size), strides=(1, 1), padding='same', pool_mode='avg')
    border = K.cast(K.greater(averaged_mask, 0.005), 'float32') * K.cast(K.less(averaged_mask, 0.995), 'float32')
    weight = K.ones_like(averaged_mask)
    w0 = K.sum(weight)
    weight += border * 2
    w1 = K.sum(weight)
    weight *= (w0 / w1)
    loss = 1 - weighted_dice_coeff(y_true, y_pred, weight)
    return loss


def weighted_bce_loss(y_true, y_pred, weight):
    # avoiding overflow
    epsilon = 1e-7
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    logit_y_pred = K.log(y_pred / (1. - y_pred))

    # https://www.tensorflow.org/api_docs/python/tf/nn/weighted_cross_entropy_with_logits
    loss = (1. - y_true) * logit_y_pred + (1. + (weight - 1.) * y_true) * \
                                          (K.log(1. + K.exp(-K.abs(logit_y_pred))) + K.maximum(-logit_y_pred, 0.))
    return K.sum(loss) / K.sum(weight)



def my_bce_loss(y_true, y_pred):
#     y_true = K.cast(y_true, 'float32')
#     y_pred = K.cast(y_pred, 'float32')
 
    # avoiding overflow
    epsilon = 1e-7
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    #logit_y_pred = K.log(y_pred / (1. - y_pred))
    loss = (y_true * K.log(y_pred[i])) + ((1 - y_true) * K.log(1 - y_pred))
    return K.mean(loss, axis=-1)        
    #return K.sum(loss)
# def my_bce_loss(y_true, y_pred):
#     y_true_f = y_true.flatten()
#     y_pred_f = y_pred.flatten()
    
#     # avoiding overflow
#     epsilon = 1e-7
#     y_pred_f[y_pred_f<=0.] = epsilon
#     y_pred_f[y_pred_f>=1.] = 1. -epsilon
#     #y_pred = K.clip(y_pred_f, epsilon, 1. - epsilon)
#     #logit_y_pred = K.log(y_pred / (1. - y_pred))
#     result = []
#     result.append([y_true[i] * math.log(y_pred[i]) + (1 - y_true[i]) * math.log(1 - y_pred[i]) \
#                    for i in range(len(y_pred))])
#     return np.mean(result)



def penalized_bce_loss(weight):
    def weighted_bce_loss(y_true, y_pred):
        # avoiding overflow
        epsilon = 1e-7
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        logit_y_pred = K.log(y_pred / (1. - y_pred))

        # https://www.tensorflow.org/api_docs/python/tf/nn/weighted_cross_entropy_with_logits
        loss = (1. - y_true) * logit_y_pred + (1. + (weight - 1.) * y_true) * (K.log(1. + K.exp(-K.abs(logit_y_pred))) + K.maximum(-logit_y_pred, 0.))
        return K.sum(loss) / K.sum(weight)
    return weighted_bce_loss

def penalized_bce_loss2(weight):
    def weighted_bce_loss(y_true, y_pred):
        # avoiding overflow
        epsilon = 1e-7
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        logit_y_pred = K.log(y_pred / (1. - y_pred))

        # https://www.tensorflow.org/api_docs/python/tf/nn/weighted_cross_entropy_with_logits
        loss = (1. - y_true) * logit_y_pred + (1. + (weight - 1.) * y_true) * \
                                              (K.log(1. + K.exp(-K.abs(logit_y_pred))) + K.maximum(-logit_y_pred, 0.))
        return K.sum(loss) / K.sum(weight)
    return weighted_bce_loss

def weighted_bce_dice_loss(y_true, y_pred):
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    # if we want to get same size of output, kernel size must be odd number
    if K.int_shape(y_pred)[1] == 128:
        kernel_size = 11
    elif K.int_shape(y_pred)[1] == 256:
        kernel_size = 21
    elif K.int_shape(y_pred)[1] == 512:
        kernel_size = 21
    elif K.int_shape(y_pred)[1] == 1024:
        kernel_size = 41
    else:
        raise ValueError('Unexpected image size')
    averaged_mask = K.pool2d(
        y_true, pool_size=(kernel_size, kernel_size), strides=(1, 1), padding='same', pool_mode='avg')
    border = K.cast(K.greater(averaged_mask, 0.005), 'float32') * K.cast(K.less(averaged_mask, 0.995), 'float32')
    weight = K.ones_like(averaged_mask)
    w0 = K.sum(weight)
    weight += border * 2
    w1 = K.sum(weight)
    weight *= (w0 / w1)
    loss = weighted_bce_loss(y_true, y_pred, weight) + (1 - weighted_dice_coeff(y_true, y_pred, weight))
    return loss

########################################
# 
# Method to evaluate the performance of u-net by computing, logloss, precision, reccall, f1score etc
#
#################################################

def evaluate_performance(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    y_pred_r = np.round(y_pred_f)

    # avoiding overflow
    epsilon = 1e-7
    y_pred_f[y_pred_f<=0.] = epsilon
    y_pred_f[y_pred_f>=1.] = 1. -epsilon
    #y_pred = K.clip(y_pred_f, epsilon, 1. - epsilon)
    #logit_y_pred = K.log(y_pred / (1. - y_pred))
    perf = {}
    result = []
    result2 = []
    
    true_p = 0.0
    true_n = 0.0
    false_p = 0.0
    false_n = 0.0
    for i in range (len(y_pred_f)):
        result.append (y_true_f[i] * np.log(y_pred_f[i]) + (1 - y_true_f[i]) * np.log(1 - y_pred_f[i]))
        result2.append (y_true_f[i] * np.log2(y_pred_f[i]) + (1 - y_true_f[i]) * np.log2(1 - y_pred_f[i]))

        if (y_pred_r[i] == 0 and y_true_f[i] == 0):
            true_n += 1.
        elif (y_pred_r[i] == 0 and y_true_f[i] == 1):
            false_n += 1.
        elif (y_pred_r[i] == 1 and y_true_f[i] == 1):
            true_p += 1.
        elif (y_pred_r[i] == 1 and y_true_f[i] == 0):
            false_p += 1.
            
    loss = np.mean(result)
    loss2 = np.mean(result2)
    accuracy = (true_p + true_n)/(true_p + true_n + false_p + false_n)
    precision = true_p/(true_p + false_p)
    recall    = true_p/(true_p + false_n)
    f1_score = (2 * precision * recall)/(precision+recall)
    
    print (len(result), sum(result))
    print ("true_pos : %d, false_pos : %d, true_neg  %d, false_neg : %d"%(true_p, false_p, true_n, false_n))
    print ("accuracy : %f, precision : %f, recall  %f, f1_score : %f"%(accuracy, precision, recall, f1_score))
    print ("logloss : %f, log2loss : %f "%(loss, loss2))
    perf["logloss"] = loss
    perf["log2loss"] = loss2
    perf["true_positive"] = true_p
    perf["false_positive"] = false_p
    perf["true_negative"] = true_n
    perf["false_negative"] = false_n
    perf["accuracy"] = accuracy
    perf["precision"] = precision
    perf["recall"] = recall
    perf["f1_score"] = f1_score
    
    return perf
