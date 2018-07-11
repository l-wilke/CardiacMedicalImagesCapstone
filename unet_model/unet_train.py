#!/usr/bin/env python

import cv2 
import re, sys
import fnmatch, shutil, subprocess
from IPython.utils import io
import glob
import random
import json
import os 

import numpy as np
from keras.models import *
from keras.layers import Input, concatenate, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
#from keras.losses import binary_crossentropy
import keras.backend as K
from keras import backend as keras
#from helper_functions  import *
from helper_utilities  import *

from loss_functions  import *
from unet_model  import *

#Fix the random seeds for numpy (this is for Keras) and for tensorflow backend to reduce the run-to-run variance
from numpy.random import seed
seed(12345)
from tensorflow import set_random_seed
set_random_seed(12345)

import tensorflow as tf

GPU_CLUSTER = "0,1,2,3,4,5,6,7" # set in config file, override later
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_CLUSTER
GPUs = None

from modelmgpu import ModelMGPU # switching between multi and the orig model object

import time

start = time.time()
print("START:", start)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

print("\nSuccessfully imported packages!!!\n")

            
#################
# Method to create a U-net model and train it
# Create a U-Net model, train the model and run the predictions and save the trained weights and predictions
#
##########################

def train_unet_model(model_name, image_size, training_images, training_labels, test_images, test_labels, model_path, batch_normalization, dropout, optimizer, learningrate, lossfun, batch_size, epochs, augmentation = False, model_summary = False):
    """
    Create a U-net model and train the model, run the predictions and save the trained weights and predictions

    Args:
      model_name: model name in string format for identification
      image_size:  176 or 256
      training_images: 4-d numpy array of images for training
      training_labels: 4-d numpy array of labels for training
      test_images:  4-d numpy array of test images
      test_labels:  4 dimensional numpy array of test labels
      model_path:  dirrectory path where the model file will reside
      dropout:  True or False
      optimizer:  Adam, etc.
      learningrate: float, 0.00001 is a good start
      lossfun:  binary_crossentropy for example 
      batch_size:  process batch size
      epochs:  80 or other int size
      augmentation:  (Default value = False), True or False
      model_summary:  (Default value = False), True or False

    Returns: class instance of myUnet which does the actual model training

    """
    global GPUs
    
    #samples, x, y, z = pred.shape
    
    train_data = {}
    test_data = {}

    train_data["images"] = training_images
    train_data["labels"] = training_labels
    test_data["images"] = test_images
    test_data["labels"] = test_labels
    
    if not os.path.exists(model_path):
        print ("creating dir ", model_path)
        os.makedirs(model_path)
            
    # get the u-net model and load train and test data
    myunet = myUnet(model_name = model_name, nGPU = GPUs, image_size = image_size, batch_norm = batch_normalization, dropout = dropout, optimizer = optimizer, lr=learningrate, loss_fn = lossfn)
    myunet.load_data(train_data, test_data)

    if (model_summary == True):
        print ("Printing model summary ")
        myunet.model.summary()
        print ("***********************************************")
        print("*************** Parallel Model summary**********")
        myunet.parallel_model.summary()
        
    res = myunet.train_and_predict(model_path, batch_size = batch_size, nb_epoch = epochs, augmentation = augmentation)
    
    return myunet


if __name__ == "__main__":
    if len(sys.argv) < 1:
        print ('Provide a config file')

    myconfig = sys.argv[1]

    if myconfig.endswith('.py'):
        myconfig = myconfig.replace('.py','')

    args = __import__(myconfig)

    img_size_list = [176, 256]
    arg_list = ['model_name','image_size','training_images','training_labels','test_images','test_labels','model_path','batch_normalization','dropout','optimizer','lr','lossfn','batch_size','epochs','augmentation','model_summary','GPU_CLUSTER','per_process_gpu_memory_fraction']
    dir_args = dir(args)

    for x in arg_list:
        if x not in dir_args:
            print ("insufficient arguments ")
            print ("enter model_name, image_size, training_images, training_labels, test_images, test_labels, model_path,batch_normalization (True or False), dropout (True or False), optimizer, learningrate, loss_function, batch_size, epochs, augmentation (True or False), model_summary (True or False) in unet_train_config")
            sys.exit() 

    image_size = args.image_size

    if image_size not in img_size_list:
        print ("image size %d is not supported"%image_size)
        sys.exit()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU_CLUSTER
    GPUs = len(args.GPU_CLUSTER.split(','))
    config.gpu_options.per_process_gpu_memory_fraction = args.per_process_gpu_memory_fraction
    session = tf.Session(config=config)

    model_name = args.model_name
    training_images = args.training_images
    training_labels = args.training_labels
    test_images = args.test_images
    test_labels = args.test_labels
    model_path = args.model_path
    batch_normalization = args.batch_normalization
    dropout = args.dropout
    optimizer = args.optimizer
    lr = args.lr
    lossfn = args.lossfn
    batch_size = args.batch_size
    epochs = args.epochs
    augmentation = args.augmentation
    model_summary = args.model_summary

    print (model_name, image_size, training_images, training_labels, test_images, test_labels, model_path,batch_normalization, dropout)
    print (optimizer, lr, lossfn, batch_size, epochs, augmentation, model_summary) 
    
    mymodel = train_unet_model(model_name, image_size, training_images, training_labels, test_images, test_labels, model_path, batch_normalization,  dropout, optimizer, lr, lossfn, batch_size, epochs, augmentation, model_summary)
    
    mymodel.save_model_info(model_path)           

    end = time.time()
    print ("END:", end - start)
