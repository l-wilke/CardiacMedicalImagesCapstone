#!/usr/bin/env python

""" Create dsb files in 4d array for testing """

import cv2 
import re, sys
import fnmatch, shutil, subprocess
import glob
import os 
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from method1 import Method1
import config

# dummy parms
config.method = 0
config.source = ''
config.path = ''
config.type = 0
m1 = Method1(config)

#Fix the random seeds for numpy (this is for Keras) and for tensorflow backend to reduce the run-to-run variance
from numpy.random import seed
seed(100)
from tensorflow import set_random_seed
set_random_seed(200)

print("\nSuccessfully imported packages!!!\n")

BASE_DIR = "/masvol/output/"
SOURCE = "dsb"
PATH = "train"
TRAIN_IMG_DIR = BASE_DIR + SOURCE + "/norm/1/3/{0}/".format(PATH)
UNET_TRAIN_DIR = BASE_DIR + SOURCE + "/norm/1/3/unet_model_{0}/data/".format(PATH)

##################################
#
# Methods to extract contour files and corresponding image files 
# and Save them as numpy arrays in memory
#
###################################

#Settings

def shrink_case(case):
    """
    Reformatting the patient id

    Args:
      case:  patient id

    Returns: patien id in the desired format

    """
    toks = case.split("-")
    
    def shrink_if_number(x):
        """
        

        Args:
          x: patient id captured

        Returns: patient id in the format we want

        """
        try:
            cvt = int(x)
            return str(cvt)
        except ValueError:
            return x

    return "-".join([shrink_if_number(t) for t in toks])

class Image_info_map(object):
    """Capturing patient id, slice, and frame ids for identification """
    def __init__(self, ctr_path):
        self.ctr_path = ctr_path
        #print (ctr_path)
        #/opt/output/dsb/norm/1/3/test/1111/sax_13_IM-2680-0008.dcm.npy
        match = re.search(r"/([^/]*)/sax_(\d+)_IM-(\d+)-(\d+(-\d+)?).dcm.npy", ctr_path)
        #match = re.search(r"/([^/]*)/patient\d+_slice(\d+)_frame(\d+)_label_fix.nii.npy", ctr_path)
        #try:
        self.case = shrink_case(match.group(1))
        self.sax_num = int (match.group(2))
        self.record = int(match.group(3))
        self.img_no = match.group(4)
            #self.img_no = int(match.group(4))
        #except AttributeError:
        #    print ('IN EXCEPT', ctr_path)
            #/opt/output/dsb/norm/1/3/train/234/sax_20_IM-3098-0022-0007.dcm.npy
        #    match = re.search(r"/([^/]*)/sax_(\d+)_IM-(\d+)-(\d+)-(\d+).dcm.npy", ctr_path)
        #    self.case = shrink_case(match.group(1))
        #    self.sax_num = int (match.group(2))
        #    self.record = int(match.group(3))
        #    self.img_no = int(match.group(5))
            
    def __str__(self):
        return "<Image info for case %s, record %d image %d>" % (self.case, self.record, self.img_no)
    
    __repr__ = __str__


def get_dsb_image_list(data_path):
    """
    Get all the dicom numpy array files and map them to the image info extracted from the regex

    Args:
      data_path: Source file path

    Returns: image list and the corresponding image info extracted

    """
    print ('DP', data_path)
    image_list = [os.path.join(data_path,f) 
            for f in glob.glob(data_path+"/*") if f.endswith('dcm.npy')]
    
    print("Number of examples: {:d}".format(len(image_list)))
    
    if len(image_list) == 0:
        print ("STOP")
        sys.exit()

    print (image_list[0], image_list[-1])
    
    extracted = list(map(Image_info_map, image_list))
    print ("Image 0 :", extracted[0].case, extracted[0].record, extracted[0].img_no)
    print ("Image -1 :", extracted[-1].case, extracted[-1].record, extracted[-1].img_no) 
    return image_list, extracted

def crop_center(img,cropx,cropy):
    """
    Function to crop the image outward from the center. 

    Args:
      img: Numpy image array
      cropx: Int value by which to crop the image in the x direction
      cropy: Int value by which to crop the image in the y direction

    Returns:
      Numpy image array with the desired crop size (cropx x cropy)

    """
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]

def get_dsb_images(img_path_list, img_count, crop_size):
    """
    Get the images and crop them to the uniform size specified

    Args:
      img_path_list:  images passed in
      img_count: The number of images passed in
      crop_size: The size of the image for cropping

    Returns: images, their file paths, and the image info

    """
    
    imgs, img_path, ext_info = [], [], []

    for i in range(img_count):
        full_path = img_path_list[i]
        img = np.load(full_path)
        #print (img.size, full_path, img.shape)
        img = m1.get_square_crop(img,crop_size,crop_size)

        """
        x,y = img.shape

        print ("File: ", full_path, x, y, crop_size)
        if x < crop_size or y < crop_size:
            #print ("shapes smaller than crop size", x, y, crop_size)
            continue

        if x > crop_size or y > crop_size:
            #print ("img: ", i, x, y, full_path)
            img = crop_center(img,crop_size,crop_size)
        """    
        imgs.append(img)
        img_path.append(full_path)
        ext_info.append(extracted_info[i])

        if i % (img_count//5) == 0:
            print (full_path)

    print ("Final size of image set :", len(imgs), "dropped images:", (img_count - len(imgs)))
                
    return imgs, img_path, ext_info

def convert_images_to_nparray_and_save (imgs, save_file, image_size):
    """
    Function to convert the dsb images for one patient into a 4-D array.

    Args:
      imgs: Images for one patient.
      save_file: Path of where to save the file
      image_size: Size of the input image

    Returns:
      Numpy 4-D array for the patient images

    """
    rows = image_size
    cols = image_size
    i = 0
    print('-'*30)
    print("Converting data to np array, Input size : ",len(imgs))
    print('-'*30)
    
    imgdatas = np.ndarray((len(imgs),rows,cols,1), dtype=np.int)
        
    for idx in range(len(imgs)):
        img = imgs[idx]
        img = img_to_array(img)        
        try:
            imgdatas[i] = img
            i += 1
        except Exception as e:
            print (e)
            continue
        
    np.save(save_file, imgdatas)

    print ("Shape of image array : ", imgdatas.shape)
    print ("Max, min, mean values", imgdatas.max(), imgdatas.min(), imgdatas.mean())
    print('Saved data as: ', save_file)


if __name__ == "__main__":
    arg = sys.argv[1:]

    print (arg)

    if len(arg) != 2:
        print ("pythong3 unet_preprocess_dsb.py 966 test")
        print ("patient id folder, and test, train, or validate")
        sys.exit()

    patient, PATH = arg[0], arg[1]

    TRAIN_IMG_DIR = BASE_DIR + SOURCE + "/norm/1/3/{0}/".format(PATH)
    UNET_TRAIN_DIR = BASE_DIR + SOURCE + "/norm/1/3/unet_model_{0}/data/".format(PATH)

    print (TRAIN_IMG_DIR,UNET_TRAIN_DIR)

    img_path_list_file = UNET_TRAIN_DIR + SOURCE + "_{0}_image_path.txt".format(patient)

    filepath = "{0}{1}".format(TRAIN_IMG_DIR,patient)

    img_path_list, extracted_info = get_dsb_image_list(filepath)
    img_count = len(img_path_list)
    print("Processing {:d} images and labels...".format(img_count))

    image_size = 176
    image_list, image_path_list, extracted_info = get_dsb_images(img_path_list, img_count, image_size)

    with open(img_path_list_file, 'w') as output:
        output.write("{0}\n".format(",".join(image_path_list)))

    if len(image_list) > 0:
        img_file = UNET_TRAIN_DIR + SOURCE + "_{0}_{1}_train.npy".format(patient, image_size)
        convert_images_to_nparray_and_save(image_list, img_file, image_size)
    else:
        print ("No Data", img_file)
        sys.exit()

    image_size2 = 256
    image_list2, image_path_list2, extracted_info2 = get_dsb_images(img_path_list, img_count, image_size2)

    ### Create 256x256 size train/test data in 4d tensor shape and save them

    if len(image_list2) > 0:
        img_file = UNET_TRAIN_DIR + SOURCE + "_{0}_{1}_train.npy".format(patient, image_size2)
        convert_images_to_nparray_and_save (image_list2, img_file, image_size2)
    else:
        print ("No Data", img_file)
        sys.exit()


