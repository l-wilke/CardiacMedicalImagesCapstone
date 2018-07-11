#!/usr/bin/env python

""" Module for predicting images files with the supplied of the model file
    Parameters are set in a config file passed in
"""

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
from keras import backend as keras
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
#from keras.losses import binary_crossentropy
import keras.backend as K
#from loss_functions  import *
from helper_utilities  import *
from unet_model import *

from modelmgpu import ModelMGPU

GPU_CLUSTER = "0,1,2,3,4,5,6,7" # set in config file, override later
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_CLUSTER
GPUs = None

#Fix the random seeds for numpy (this is for Keras) and for tensorflow backend to reduce the run-to-run variance
from numpy.random import seed
seed(100)
from tensorflow import set_random_seed
set_random_seed(200)

import tensorflow as tf

import time
start = time.time()
print("START:", start)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

print("\nSuccessfully imported packages!!!\n")


def predict_with_pretrained_weights(args):
    """
    Load the pretrained weights with the model file to predict
    
    Args:
      args: parameters passed in from the prediction config file

    Returns: none

    """
    print ("Creating U-net model...")
    myunet = myUnet(model_name = args.model_name, nGPU = args.GPUs, image_size = args.image_size, batch_norm = args.batch_normalization, dropout = args.dropout)
    print('-'*30)
    myunet.model_file = args.hdf5_path
    print ("Loading the pre-trained weights...", myunet.model_file)
    myunet.load_pretrained_weights(myunet.model_file)
    print('-'*30)
 
    myunet.model_path = "{0}_models".format(args.predict_path) # where one_count.json files are
    myunet.volume_path = args.volume_path
    myunet.source_type = args.source_type # train, test, validate, or roi
    myunet.source = args.source # train, test, validate
    myunet.data_source = "unet_model_{0}".format(myunet.source_type)
    myunet.file_source = args.file_source
    myunet.test_source_path = "/masvol/output/{0}/norm/{1}/{2}".format(myunet.file_source, args.method, args.Type)
    myunet.method = args.method
    myunet.image_size = args.image_size
    myunet.predict_path = "{0}_predict".format(args.predict_path) # where ts_norm, pred_round, and predictions.json reside
    myunet.augmentation = args.augmentation
    myunet.batch_norm = args.batch_normalization
    myunet.dropout = args.dropout
    myunet.contrast_normalize = args.contrast_normalization

    inputpath = "/masvol/data/{0}/{1}/*".format(myunet.file_source, myunet.source)
    pcount = 0
    print (inputpath)
    topdir = "{0}/{1}".format(myunet.test_source_path, myunet.data_source)

    for i in glob.glob(inputpath):

        nodes = i.split('/')
        myunet.patient = nodes[-1]
        print ('|'+myunet.patient+'|')

        if len(args.patient_list) > 0:
            patient = int(myunet.patient)

            if patient not in args.patient_list:
                continue

        # test_image_file: image files in 4d numpy array for testing, image_4d_file
        myunet.image_4d_file = "{0}/data/{1}_{2}_{3}_train.npy".format(topdir, myunet.file_source, myunet.patient, myunet.image_size)
        print ('4d',myunet.image_4d_file)
        myunet.image_source_file = "{0}/data/{1}_{2}_image_path.txt".format(topdir, myunet.file_source, myunet.patient)
        print ('sf',myunet.image_source_file)
        myunet.image_one_file = "{0}/{1}/{2}_{3}_{4}_one_count.json".format(topdir, myunet.model_path, myunet.file_source, myunet.patient, myunet.image_size)
        # test_labels:  (Default value = "none"), label files or none
        myunet.test_labels = "none"
        #print ('io',myunet.image_one_file)
        pred_file = "{0}/{1}/{2}_{3}_{4}_predictions.npy".format(topdir,myunet.predict_path,myunet.file_source,myunet.patient,myunet.image_size)
        print ('pred_file', pred_file)
        myunet.do_predict()

        pred_dir = os.path.dirname(pred_file)
        print ('pred_dir', pred_dir)

        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)

        np.save(pred_file, myunet.predictions)
        myunet.pred_round = np.round(myunet.predictions)
        pred_round_file = "{0}/{1}_{2}_{3}_pred_round.npy".format(pred_dir,myunet.file_source,myunet.patient,myunet.image_size)
        np.save(pred_round_file, myunet.pred_round)

        ts_norm_file = "{0}/{1}_{2}_{3}_ts_norm.npy".format(pred_dir,myunet.file_source,myunet.patient,myunet.image_size)
        np.save(ts_norm_file, myunet.test_images)

        predictions = remove_ctrsAll(myunet.pred_round)  # remove on rounded

        pred_file_CR = "{0}/{1}_{2}_{3}_NCR1_predictions_cleaned.npy".format(pred_dir, myunet.file_source, myunet.patient, myunet.image_size)
        np.save(pred_file_CR, predictions)                     

        sourcedict = get_ones(predictions, myunet.image_source_file)
        image_dir = os.path.dirname(myunet.image_one_file)

        if not os.path.exists(image_dir):
            os.makedirs(image_dir)

        with open(myunet.image_one_file, 'w') as output:
            output.write("{0}\n".format(json.dumps(sourcedict)))

        origpath = '/masvol/data/{0}/{1}/{2}/study/'.format(myunet.file_source,myunet.source,str(myunet.patient))
        newpath = '/masvol/output/{0}/volume/{1}/{2}/{3}_{4}_{5}.json'.format(myunet.file_source,myunet.method,myunet.volume_path,myunet.source,str(myunet.patient),myunet.image_size)
        dump_and_sort(myunet.image_one_file, origpath, newpath)

        pcount += 1
        #if pcount > 1:
        #    break


if __name__ == "__main__":
    if len(sys.argv) < 1:
        print ('Provide a config file')

    myconfig = sys.argv[1]

    if myconfig.endswith('.py'):
        myconfig = myconfig.replace('.py','')

    args = __import__(myconfig)

    img_size_list = [176, 256]

    # fit and predict for all patients with source_type
    # model_name:  unique model name for identification
    # GPUs: number of GPU devices using
    # model_file: model file for predicting, hdf5_path
    # image_size: uniform image size
    # augmentation:  (Default value = False)
    # contrast_normalization: (Default value = False)
    arg_list = ['model_name','image_size','source','source_type','method','Type','hdf5_path','predict_path','volume_path','augmentation','contrast_normalization','batch_normalization','GPU_CLUSTER','GPUs','per_process_gpu_memory_fraction','file_source','dropout','patient_list']

    dir_args = dir(args)

    for x in arg_list:
        if x not in dir_args:
            print ("insufficient arguments ")
            print ("enter {0} in the config file".format(",".join(arg_list)))
            sys.exit() 

    if args.image_size not in img_size_list:
        print ("image size %d is not supported"%args.image_size)
        sys.exit()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU_CLUSTER
    args.GPUs = args.GPUs
    config.gpu_options.per_process_gpu_memory_fraction = args.per_process_gpu_memory_fraction
    session = tf.Session(config=config)

    predict_with_pretrained_weights(args)

    end = time.time()
    print ("END:", end - start)
