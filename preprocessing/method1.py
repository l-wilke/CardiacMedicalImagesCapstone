#!/usr/bin/env python

import glob
import sys
import dicom
import os
import numpy as np
import glob
import cv2 
import preproc
import nibabel as nib
from helpers_dicom import DicomWrapper as dicomwrapper


class Method1(object):
    """Normalization method 1 """
    def __init__(self, arg):
            self.arg = arg
            self.method = self.arg.method
            self.path = self.arg.path
            self.source = self.arg.source
            self.sourceinfo = None
            self.inputfiles = None
            self.type = int(self.arg.type)
            self.filesource = dict()

    def get_config(self):
        """Making sure we are applying normalization for method1"""

        if self.method != 1 and self.method != '1':
            print ("method 1")
            sys.exit()

        if self.source not in preproc.sources.keys():
            sys.exit()

    def main_process(self):
        """Main process function to apply normalization on different data sources """
        self.get_config()
        self.get_init_path()

        if self.source == 'dsb':
            self.update_filesource('source', self.source)
            self.get_dsb_files()
            return

        if self.source == 'sunnybrook':
            self.update_filesource('source', self.source)

            if self.path == 'challenge':
                self.get_sunnybrook_files2()
                return

            self.get_sunnybrook_files()
            return

        if self.source == 'acdc':
            self.update_filesource('source', self.source)
            self.get_acdc_files()
            return

    def update_filesource(self, key, value, append=0):
        """
        Function that updates the filesource dictionary for acdc, sunnybrook, or dsb, to identify which images are getting normalized. 
        Not being used, but it's available

        Args:
          key: Dictionary key for what we want to store, but not used currently 
          value: Dictionary value for what we want to store, but not used currently
          append:  (Default value = 0), if the value is in the list format, append flag can be turned on to append

        Returns: none

        """
        #print 'k', key, 'v', value, 'a', append

        if key not in self.filesource:
            if append:
                self.filesource.update({key: [value]})
                return

            self.filesource.update({key: value})
            return

        elif append:
            self.filesource[key].append(value)
            return
              
        self.filesource.update({key: value})

    def get_init_path(self):
        """Get data path of the source """
        self.sourceinfo = preproc.sources[self.source]

        if self.path not in self.sourceinfo['paths']:
            print ("valid paths {0}".format(self.sourceinfo['paths']))
            sys.exit() 

        path = "{0}/{1}/*".format(self.sourceinfo['dir'], self.path)
        self.update_filesource('path', path)
        self.inputfiles = glob.glob(path)

    def orientation_flip180(self, img):
        """
        Function that fips the image by 180 degrees 
        
        Args:
          img: img in numpy array for flipping horizontally and vertically

        Returns: 
          img in numpy array flipped

        """
        return cv2.flip(img,-1)

    def get_acdc_files(self):
        """Get ACDC source data then check for file and processing types to apply normalization"""

        for f in self.inputfiles:
            nodes = f.split('/')

            inputs = glob.glob("{0}/{1}/{2}*".format(f,self.sourceinfo['string'],self.sourceinfo['pattern']))
            print ('inputs', inputs)

            patient = None

            for i in inputs:
                print ('i', i)

                if i.endswith('labels'):
                    for root, _, files in os.walk(i):
                        rootnode = root.split("/")
                        label = rootnode[-1]
                        tempdir = i.replace('/'+label,'')
                        patient = rootnode[-2]
                        print ('rootnode', rootnode, label, patient, files, tempdir)

                        for f in files:
                            nim1dir = root.replace('/'+label,'')
                            nim1label = nib.load(root+'/'+f)
                            spacing = nim1label.header.get('pixdim')[1:3]
                            #spacing = nim1label.header.get('pixdim')[1]
                            flippedlabel = self.orientation_flip180(nim1label.get_data())
                            norm = None

                            if self.type == 0 or self.type == '0':
                                norm = self.original_method_acdc(flippedlabel, spacing)
                            elif self.type == 1 or self.type == '1':
                                norm = self.new_rescaling_method_acdc(flippedlabel, spacing, 1)
                            elif self.type == 3 or self.type == '3':
                                norm = self.rescaling_only_method_acdc(flippedlabel, spacing)

                            outfilename = "{0}.npy".format(f)
                            outpath = "{0}/{1}/{2}/{3}".format(preproc.normoutputs[self.source]['dir'], self.method, self.type, patient)

                            if not os.path.isdir(outpath):
                                os.mkdir(outpath)

                            if norm is not None:
                                np.save("{0}/{1}".format(outpath, outfilename), norm)

                            outfilenamenodes = outfilename.split('_')
                            slicepath = "{0}_{1}".format(outfilenamenodes[0], outfilenamenodes[1])
                            slicedir = "{0}/{1}".format(nim1dir, slicepath)

                            for root2, _, files2 in os.walk(slicedir):
                                for f2 in files2:
                                    nim1 = nib.load(root2+'/'+f2)
                                    spacing2 = nim1.header.get('pixdim')[1:3]
                                    #spacing2 = nim1.header.get('pixdim')[1]
                                    flipped = self.orientation_flip180(nim1.get_data())

                                    norm2 = None

                                    if self.type == 0 or self.type == '0':
                                        norm2 = self.original_method_acdc(flipped, spacing2)
                                    elif self.type == 1 or self.type == '1':
                                        norm2 = self.new_rescaling_method_acdc(flipped, spacing2)
                                    elif self.type == 3 or self.type == '3':
                                        norm2 = self.rescaling_only_method_acdc(flipped, spacing2)

                                    outfilename2 = "{0}.npy".format(f2)

                                    if norm2 is not None:
                                        np.save("{0}/{1}".format(outpath, outfilename2), norm2)

    def get_sunnybrook_files2(self):
        """Applying normalization on the sunnybrook image and label files """

        for i in self.inputfiles:
            if i.endswith('pdf'):
                continue

            for root, _, files in os.walk(i):
                rootnode = root.split("/")[-1] # sax file

                newroot = root.replace(self.path, '*')
                newfile = None

                for f in files:
                    if f.endswith('.dcm.img.npy'):

                        newfile = f.replace('.img.npy', '')
                        lblfile = f.replace('.img.npy', '.label.npy')

                        for nfile in glob.glob("{0}/{1}".format(newroot,newfile)):
                            nodes = nfile.split('/')
                            nroot = "/".join(nodes[:-1])
                            dw = None
                            norm = None

                            dw = dicomwrapper(nroot+'/', newfile)
                            npyload = np.load("{0}/{1}".format(i,lblfile))

                            if self.type == 0 or self.type == '0':
                                norm = self.original_method_npy(dw, npyload, 1)
                            elif self.type == 1 or self.type == '1':
                                norm = self.new_rescaling_method_acdc(npyload, dw.spacing, 1)
                            elif self.type == 2 or self.type == '2':
                                norm = self.no_orientation_npy(npyload, dw.spacing, 1)
                            elif self.type == 3 or self.type == '3':
                                norm = self.rescaling_only_method_acdc(npyload, dw.spacing)

                            outfilename = lblfile
                            outpath =  "{0}/{1}/{2}/{3}/{4}".format(preproc.normoutputs[self.source]['dir'], self.method, self.type, self.path, rootnode)

                            if not os.path.isdir(outpath):
                                os.mkdir(outpath)

                            if norm is not None:
                                np.save("{0}/{1}".format(outpath, outfilename), norm)

    def get_sunnybrook_files(self):
        """Converting the sunnybrook dicom files to numpy array files with normalization """

        for i in self.inputfiles:
            if i.endswith('pdf'):
                continue

            patientslices = dict()

            for root, _, files in os.walk(i):
                rootnode = root.split("/")[-1] # sax file
                patientslices.update({root: []})

                for f in files:
                    if not f.endswith('.dcm'):
                        continue

                    dw = None

                    try:
                        dw = dicomwrapper(root+'/', f)
                    except:
                        continue

                    """
                    patientframe = dict()
                    patientframe.update({'filename': f})
                    patientframe.update({'InPlanePhaseEncodingDirection': dw.in_plane_encoding_direction})

                    patientslices.update({'image_position_patient': dw.image_position_patient})
                    patientslices.update({'image_orientation_patient': dw.image_orientation_patient})
                    patientslices.update({'PixelSpacing': dw.spacing})
                    #patientslices.update({'PatientAge': dw.PatientAge})

                    patientslices[root].append(patientframe)
                    """
                    norm = None

                    if self.type == 0 or self.type == '0':
                        norm = self.original_method(dw, 1) # 1 for yes, convert to unicode int 16
                    elif self.type == 1 or self.type == '1':
                        norm = self.new_rescaling_method(dw)
                    elif self.type == 2 or self.type == '2':
                        norm = self.no_orientation_method(dw)
                    elif self.type == 3 or self.type == '3':
                        norm = self.rescaling_only_method(dw)

                    outfilename = "{0}.npy".format(f)
                    outpath =  "{0}/{1}/{2}/{3}/{4}".format(preproc.normoutputs[self.source]['dir'], self.method, self.type, self.path, rootnode)

                    if not os.path.isdir(outpath):
                        os.mkdir(outpath)

                    if norm is not None:
                        np.save("{0}/{1}".format(outpath, outfilename), norm)

    def get_dsb_files(self):
        """Get the data science bowl files to apply normalization """

        for f in self.inputfiles:
            #print 'f', f
            nodes = f.split('/')

            patient = int(nodes[-1])

            inputs = glob.glob("{0}/{1}/{2}*".format(f,self.sourceinfo['string'],self.sourceinfo['pattern']))

            for i in inputs:
                patientslices = dict()

                for root, _, files in os.walk(i):

                    rootnode = root.split("/")[-1] # sax file
                    patientslices.update({root: []})

                    for f in files:
                        if not f.endswith('.dcm'):
                            continue

                        dw = dicomwrapper(root+'/', f)

                        if int(dw.patient_id) != patient:
                            print ('Error')
                            sys.exit()

                        """
                        patientframe = dict()
                        patientframe.update({'filename': f})
                        patientframe.update({'InPlanePhaseEncodingDirection': dw.in_plane_encoding_direction})

                        patientslices.update({'image_position_patient': dw.image_position_patient})
                        patientslices.update({'image_orientation_patient': dw.image_orientation_patient})
                        patientslices.update({'PixelSpacing': dw.spacing})
                        patientslices.update({'PatientAge': dw.PatientAge})

                        patientslices[root].append(patientframe)
                        """

                        norm = None

                        if self.type == 0 or self.type == '0':
                            norm = self.original_method(dw)
                        elif self.type == 1 or self.type == '1':
                            norm = self.new_rescaling_method(dw)
                        elif self.type == 2 or self.type == '2':
                            norm = self.no_orientation_method(dw)
                        elif self.type == 3 or self.type == '3':
                            norm = self.rescaling_only_method(dw)

                        outfilename = "{0}_{1}.npy".format(rootnode, f)
                        outpath =  "{0}/{1}/{2}/{3}/{4}".format(preproc.normoutputs[self.source]['dir'], self.method, self.type, self.path, dw.patient_id)

                        if not os.path.isdir(outpath):
                            os.mkdir(outpath)

                        if norm is not None:
                            np.save("{0}/{1}".format(outpath, outfilename), norm)

                #self.update_filesource(patient, {'patientfiles':patientslices}, 1)

    #Original method npy
    def original_method_npy(self, dw, npyarray, convert):
        """
        The intial normalization method (rotating if plane in column, rescaling, cropping, and applying contrast) before other methods and types existed 

        Args:
          dw: dicome wrapper object instantiated from helpers dicom wrapper borrowed from Julian (3rd place)
          npyarray: label in numpy array
          convert: flag indicator for dtype conversion

        Returns: 
          numpy array with contrast applied

        """
        nimg = None

        if dw.in_plane_encoding_direction == 'COL':
            nimg = cv2.transpose(npyarray)
        else:
            nimg = npyarray

        rescaled = self.reScale(nimg, dw.spacing[0])
        cropped = self.get_square_crop(rescaled)

        if convert:
            converted = np.array(cropped, dtype=np.uint16)
            return self.CLAHEContrastNorm(converted)

        return self.CLAHEContrastNorm(cropped)

    #Original method acdc
    def original_method_acdc(self, img, spacing):
        """
        Function that calls reScale, get_square_crop, and CLAHE for the ACDC images. 

        Args:
          img: numpy image array
          spacing: Pixel spacing tuple

        Returns:
          Numpy image array that has been normalized.

        """
        rescaled = self.reScale(img, spacing)
        cropped = self.get_square_crop(rescaled)
        converted = np.array(cropped, dtype=np.uint16)
        return self.CLAHEContrastNorm(converted)

    #New Rescaling acdc and sunnybrook
    def new_rescaling_method_acdc(self, img, spacing, label=0):
        """
        Function that calls reScaleNew, get_square_crop, and CLAHE for the ACDC images.
        Function that calls reScaleNew and get_square_crop for the ACDC labels.

        Args:
          img: numpy image array
          spacing: Pixel spacing tuple
          label:  (Default value = 0) Indicates if the image is a label

        Returns:
          numpy image or  label array that has been normalized.

        """
        rescaled = self.reScaleNew(img, spacing)
        cropped = self.get_square_crop(rescaled)

        if label:
            return cropped

        converted = np.array(cropped, dtype=np.uint16)
        return self.CLAHEContrastNorm(converted)

    #Rescaling only acdc and sunnybrook
    def rescaling_only_method_acdc(self, img, spacing):
        """
        Function to rescale the ACDC Images

        Args:
          img: numpy image array
          spacing: Pixel Spacing tuple

        Returns:
          numpy image array that has been rescaled using the spacing in the x and y directions

        """
        return self.reScaleNew(img, spacing)

    #Original method
    def original_method(self, dw, convert=0):
        """
        Fuction that calls InPlanePhaseEncoding, reScale, get_square_crop, and CLHAEContrastNorm.

        Args:
          dw: Dicom Image Object
          convert:  (Default value = 0)Value to idenfity if the image needs to be converted to np.unit16

        Returns:
          Numpy image array that has undergone the normalization methods. 

        """
        img = self.InPlanePhaseEncoding(dw.raw_file)
        rescaled = self.reScale(img, dw.spacing[0])
        cropped = self.get_square_crop(rescaled)

        if convert:
            converted = np.array(cropped, dtype=np.uint16)
            return self.CLAHEContrastNorm(converted)

        return self.CLAHEContrastNorm(cropped)

    #New Rescaling
    def new_rescaling_method(self, dw):
        """
        Function that calls InPlanePhaseEncoding, reScaleNew, get_square_crop, and CLAHEContrastNorm functions to apply these functions on the image. 

        Args:
          dw: Dicom Image Object

        Returns:
          Numpy Image array that has been normalized with the functions above. 
    
        """
        img = self.InPlanePhaseEncoding(dw.raw_file)
        rescaled = self.reScaleNew(img, dw.spacing)
        cropped = self.get_square_crop(rescaled)
        converted = np.array(cropped, dtype=np.uint16)
        return self.CLAHEContrastNorm(converted)

    #No Orientation npy
    def no_orientation_npy(self, img, spacing, label):
        """
        Function that called reScaleNew, get_square_crop, and CLAHEContrastNorm. The label image will be cropped and rescaled, the label will not have CLAHE applied to it. 

        Args:
          img: numpy image aray
          spacing: Pixel Spacing desired for rescaling
          label: Label image that corresponds with the image.

        Returns:
          The numpy image array with the normalization methods applied and the label that has been normalized. 
        """
        rescaled = self.reScaleNew(img, spacing)
        cropped = self.get_square_crop(rescaled)

        if label:
            return cropped

        converted = np.array(cropped, dtype=np.uint16)
        return self.CLAHEContrastNorm(converted)

    #No Orientation
    def no_orientation_method(self, dw):
        """
        Function that called reScaleNew, get_square_crop, and CLAHEContrastNorm to apply these preprocessing functions on the image.

        Args:
          dw: Dicom Image Object

        Returns:
          Numpy Image array that has been rescaled, cropped and had CLAHE applied to it. 

        """
        img = dw.raw_file
        rescaled = self.reScaleNew(img.pixel_array, img.PixelSpacing)
        cropped = self.get_square_crop(rescaled)
        converted = np.array(cropped, dtype=np.uint16)
        return self.CLAHEContrastNorm(converted)

    #Rescaling only
    def rescaling_only_method(self, dw):
        """
        Function to rescale the image. Calls the reScaleNew function with the dicom pixel array and the dicom pixel spacing as inputs.

        Args:
          dw: Dicom image object

        Returns:
          Numpy Image array that has been rescaled.
        """
        img = dw.raw_file
        return self.reScaleNew(img.pixel_array, img.PixelSpacing)

    #Function that uses the InPlanephaseEncoding to determine if COL or ROW based and then transposes and flips the image. 
    def InPlanePhaseEncoding (self, img):
        """
        Function to transpose the image and flip it based on the InPlanePhaseEncoding dicom metatdata. If the InPlanePhaseEncoding Direction is COL, then the image will be flipped, if it is ROW then the image is returned. 

        Args:
          img: dicom image

        Returns:
          numpy image array that has been flipped based on the InPlanePhaseEncodingDirection value.

        """
        if img.InPlanePhaseEncodingDirection == 'COL':
            new_img = cv2.transpose(img.pixel_array)
            #py.imshow(img_new)
            new_img = cv2.flip(new_img, 0)
            return new_img
        else:
            #print 'Row Oriented'
            return img.pixel_array

    #Function of Rescaling the pixels
    def reScale(self, img, scale):
        """
        Function to rescale the image with the same value in each direction

        Args:
          img: numpy image array
          scale: int value to rescale the image in the x and y direction

        Returns:
          Numpy array image with the pixel spacing rescaled

        """
        return cv2.resize(img, (0, 0), fx=scale, fy=scale)

    def reScaleNew(self, img, scale):
        """
        Function to rescale the image in order to make the pixel spacing 1 mm by 1 mm.

        Args:
          img: numpy image array
          scale: list of two values with the scaling in the x direction and y direction

        Returns:
          Numpy array image with the pixel spacing 1 mm by 1 mm

        """
        return cv2.resize(img, (0, 0), fx=scale[0], fy=scale[1])

    #Function to crop the image into a square
    def get_square_crop(self, img, base_size=256, crop_size=256):
        """
        Function to crop the image from the center outward or add a border to get to the desired spot. 

        Args:
          img: numpy image array
          base_size:  (Default value = 256) The size to make the image if a border is needed to get to the desired size. 
          crop_size:  (Default value = 256) The desired size to crop the image

        Returns:
          Numpy image array of the desired size. 

        """
        res = img
#         print (res)
#         print (res.shape)
        height, width = res.shape

        if height < base_size:
            diff = base_size - height
            extend_top = diff // 2
            extend_bottom = diff - extend_top
            res = cv2.copyMakeBorder(res, extend_top, extend_bottom, 0, 0, 
                                     borderType=cv2.BORDER_CONSTANT, value=0)
            height = base_size

        if width < base_size:
            diff = base_size - width
            extend_top = diff // 2
            extend_bottom = diff - extend_top
            res = cv2.copyMakeBorder(res, 0, 0, extend_top, extend_bottom, 
                                     borderType=cv2.BORDER_CONSTANT, value=0)
            width = base_size

        crop_y_start = (height - crop_size) // 2
        crop_x_start = (width - crop_size) // 2
        res = res[crop_y_start:(crop_y_start + crop_size), crop_x_start:(crop_x_start + crop_size)]
#         print (res)
#         print (res.shape)
        return res

    #Contrast Normalizaiton
    def CLAHEContrastNorm(self, img, tile_size=(1,1)):
        """
        Function to apply CLAHE normalization to the image.

        Args:
          img: numpy image array
          tile_size:  (Default value = (1,1)) The square dimentions of the tiles that the image will be broken up into. The CLAHE will be applied to each tile. 

        Returns: 
          The image with CLAHE applied.

        """
        clahe = cv2.createCLAHE(tileGridSize=tile_size)
        return clahe.apply(img)


if __name__ == "__main__":
    """For testing purpose, use config to pass in the parms for Method1 class"""

    import config
    method = Method1(config)
    method.main_process()
