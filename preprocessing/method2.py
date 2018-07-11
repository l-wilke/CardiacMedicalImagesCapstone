#!/usr/bin/env python

import glob
import sys
import dicom
import os
import numpy as np
import glob
import cv2 
from scipy.misc import imrotate
import preproc
import nibabel as nib
from helpers_dicom import DicomWrapper as dicomwrapper


class Method2(object):
    """ Normalization method 2 """
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
        """Making sure we are applying normalization for method2 """
        if self.method != 2 and self.method != '2':
            print ("method 2")
            sys.exit()

        if self.source not in preproc.sources.keys():
            sys.exit()

    """ main function """
    def main_process(self):
        """Main process function to apply normalization on different data sources """
        self.get_config()
        self.get_init_path()

        if self.source == 'dsb':
            self.update_filesource('source', self.source)
            self.get_dsb_files()
            print (self.filesource)
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
        Function that updates the filesource, acdc, sunnybrook, or dsb, to identify which images are getting normalized. 

        Args:
          key: 
          value: 
          append:  (Default value = 0)

        Returns:

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
        """Get data path of the source  """
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

    def contrast(self, img):
        """
        Function to normalize the pixel values within the image array

        Args:
          img: numpy image array

        Returns:
          image with the pixel value normalization

        """
        im_max=np.percentile(img.flatten(),99)
        return np.array(np.clip(np.array(img,dtype=np.float)/im_max*256,0,255),dtype=np.uint8)

    def get_dsb_files(self):
        """Get the data science bowl files to apply normalization """
        for f in self.inputfiles:
            print ('f', f)
            nodes = f.split('/')

            patient = int(nodes[-1])

            print (self.sourceinfo)
            inputs = glob.glob("{0}/{1}/{2}*".format(f,self.sourceinfo['string'],self.sourceinfo['pattern']))
            print ('inputs', inputs)

            for i in inputs:
                patientslices = dict()

                for root, _, files in os.walk(i):
                    rootnode = root.split("/")[-1] # sax file
                    patientslices.update({root: []})

                    for f in files:
                        if not f.endswith('.dcm'):
                            continue

                        print (root, f)
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

                        """
                        img = self.getAlignImg(dw)
                        cropped = self.crop_size(img)
                        """
                        outfilename = "{0}_{1}.npy".format(rootnode, f)
                        outpath = "{0}/{1}/{2}/{3}/{4}".format(preproc.normoutputs[self.source]['dir'], self.method, self.type, self.path, dw.patient_id)

                        if not os.path.isdir(outpath):
                            os.mkdir(outpath)

                        np.save("{0}/{1}".format(outpath, outfilename), norm)

                self.update_filesource(patient, {'patientfiles':patientslices}, 1)

    def get_acdc_files(self):
        """ Get ACDC source data then check for file and processing types"""
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
                            flippedlabel = self.orientation_flip180(nim1label.get_data())
                            
                            norm = None
                            
                            if self.type == 0 or self.type == '0':
                                #contrast = self.contrast(flippedlabel)
                                norm = self.crop_size(flippedlabel)
                            elif self.type == 1 or self.type == '1' or self.type == 3 or self.type == '3':

                                norm = self.new_rescaling_method_acdc(flippedlabel, spacing, 1)

                            outfilename = "{0}.npy".format(f)
                            outpath = "{0}/{1}/{2}/{3}".format(preproc.normoutputs[self.source]['dir'], self.method, self.type, patient)

                            if not os.path.isdir(outpath):
                                os.mkdir(outpath)

                            np.save("{0}/{1}".format(outpath, outfilename), norm)

                            outfilenamenodes = outfilename.split('_')
                            slicepath = "{0}_{1}".format(outfilenamenodes[0], outfilenamenodes[1])
                            slicedir = "{0}/{1}".format(nim1dir, slicepath)

                            for root2, _, files2 in os.walk(slicedir):
                                for f2 in files2:
                                    nim1 = nib.load(root2+'/'+f2)
                                    spacing2 = nim1.header.get('pixdim')[1:3]
                                    flipped = self.orientation_flip180(nim1.get_data())

                                    norm2 = None

                                    if self.type == 0 or self.type == '0':
                                        contrast = self.contrast(flipped)
                                        norm2 = self.croped_size(contrast)
                                    elif self.type == 1 or self.type == '1':
                                        norm2 = self.new_rescaling_method_acdc(flipped,spacing2)
                                    elif self.type == 3 or self.type == '3':
                                        norm2 = self.reScaleNew(flipped, spacing2)

                                    outfilename2 = "{0}.npy".format(f2)
                                    np.save("{0}/{1}".format(outpath, outfilename2), norm2)

    def get_sunnybrook_files2(self):
        """Applying normalization on the sunnybrook image and label files """
        for i in self.inputfiles:
            if i.endswith('pdf'):
                continue
            print (i)

            for root, _, files in os.walk(i):
                rootnode = root.split("/")[-1] # sax file
                print (rootnode, files)

                newroot = root.replace(self.path, '*')
                newfile = None

                for f in files:
                    if f.endswith('.dcm.img.npy'):
                        print ('True')

                        newfile = f.replace('.img.npy', '')
                        lblfile = f.replace('.img.npy', '.label.npy')
                        print ("{0}/{1}".format(newroot,newfile))

                        for nfile in glob.glob("{0}/{1}".format(newroot,newfile)):
                            print (nfile)
                            nodes = nfile.split('/')
                            nroot = "/".join(nodes[:-1])
                            norm = None

                            dw = dicomwrapper(nroot+'/', newfile)
                            npyload = np.load("{0}/{1}".format(i,lblfile))

                            if self.type == 0 or self.type == '0':
                                norm = self.original_method(dw, npyload)
                            elif self.type == 1 or self.type == '1':
                                norm = self.new_rescaling_method(dw, npyload)
                            elif self.type == 2 or self.type == '2':
                                norm = self.no_orientation_npy(npyload, dw.spacing, 1)
                            elif self.type == 3 or self.type == '3':
                                norm = self.rescaling_only_method_acdc(npyload, dw.spacing)

                            outfilename = lblfile
                            outpath =  "{0}/{1}/{2}/{3}/{4}".format(preproc.normoutputs[self.source]['dir'], self.method, self.type, self.path, rootnode)
                            print (outpath)
                            if not os.path.isdir(outpath):
                                os.mkdir(outpath)

                            if norm is not None:
                                np.save("{0}/{1}".format(outpath, outfilename), norm)

    def get_sunnybrook_files(self):
        """ Converting the sunnybrook dicom files to numpy array files with normalization """
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
                        print (root, f)
                        img=dicom.read_file(root+'/'+f)
                        print (img.BitsStored, img.BitsAllocated)
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
                        norm = self.original_method(dw)
                    elif self.type == 1 or self.type == '1':
                        norm = self.new_rescaling_method(dw)
                    elif self.type == 2 or self.type == '2':
                        norm = self.no_orientation_method(dw)
                    elif self.type == 3 or self.type == '3':
                        norm = self.rescaling_only_method(dw)
                    """
                    img = self.getAlignImg(dw)
                    cropped = self.crop_size(img)
                    """
                    outfilename = "{0}.npy".format(f)
                    outpath =  "{0}/{1}/{2}/{3}/{4}".format(preproc.normoutputs[self.source]['dir'], self.method, self.type, self.path, rootnode)

                    if not os.path.isdir(outpath):
                        os.mkdir(outpath)

                    np.save("{0}/{1}".format(outpath, outfilename), norm)

    def original_method(self, dw, label=None):
        """
        Function that clals getAlignImg and crop_size.

        Args:
          dw: Dicom Image Object
          label:  (Default value = None) Value to indicate if the image is a label

        Returns:
          numpy image array that has been rotate and cropped

        """
        img = self.getAlignImg(dw, label)
        cropped = self.crop_size(img)
        return cropped

    def new_rescaling_method(self, dw, label=None):
        """
        Function that calls getAlignImg, reScaleNew, and crop_size.

        Args:
          dw: Dicom Image Object
          label:  (Default value = None) Value to indicate if the image is a label

        Returns:
          numpy image or label array that has been rotated, rescaled based on pixel spacing, and cropped.

        """
        img = self.getAlignImg(dw, label)
        rescaled = self.reScaleNew(img, dw.spacing)
        return self.crop_size(rescaled)

    #No Orientation npy
    def no_orientation_npy(self, img, spacing, label):
        """
        Function that calls reScaleNew, and crop_size and contrast. 

        Args:
          img: numpy image array
          spacing: list of two values, to reScale the image by in the x and y direction
          label: indicates if the image is a label

        Returns:
          numpy image or label array that has been rescaled and cropped. 

        """
        print (img)
        rescaled = self.reScaleNew(img, spacing)
        cropped = self.crop_size(rescaled)

        if label:
            return cropped

        return self.contrast(cropped)

    def no_orientation_method(self, dw):
        """
        Function that calls reScaleNew, contrast and crop_size. 

        Args:
          dw: Dicom image object

        Returns:
          numpy image array that has been rescaled and cropped. 

        """
        img = dw.raw_file
        rescaled = self.reScaleNew(img.pixel_array, img.PixelSpacing)
        contrast = self.contrast(rescaled)
        return self.crop_size(contrast)

    def rescaling_only_method(self, dw):
        """
        Function that calls reScaleNew.

        Args:
          dw: Dicome Image array

        Returns:
          numpy image array that has been rescaled. 

        """
        img = dw.raw_file
        return self.reScaleNew(img.pixel_array, img.PixelSpacing)

    #Rescaling only acdc and sunnybrook
    def rescaling_only_method_acdc(self, img, spacing):
        """
        Function that calls reScaleNew for ACDC images

        Args:
          img: numpy image array
          spacing: spacing list that indicates what values to rescale the image by

        Returns:
          numpy image array that has been rescaled.

        """
        return self.reScaleNew(img, spacing)

    def new_rescaling_method_acdc(self, img, spacing, label=0):
        """
        Function that called reScaleNew, crop_size, and contrast.

        Args:
          img: numpy image array
          spacing: spacing list that indicates what values to rescale the image by
          label:  (Default value = 0) Value to indicate if the image is a label

        Returns:
          numpy image or label that has been normalized

        """
        rescaled = self.reScaleNew(img, spacing)
        cropped = self.crop_size(rescaled)

        if label:
            return cropped

        return self.contrast(cropped)

    def getAlignImg(self, img, label = None):#!!!notice, only take uint8 type for the imrotate function!!!
        """
        Function that rotates the image or label.

        Args:
          img: numpy image array
          label:  (Default value = None) Value to indicate if the image is a label

        Returns:
          numpy image array that has been rotated. 

        """
        f = lambda x:np.asarray([float(a) for a in x]);
        o = f(img.image_orientation_patient);
        o1 = o[:3];
        o2 = o[3:];
        oh = np.cross(o1,o2);
        or1 = np.asarray([0.6,0.6,-0.2]);
        o2new = np.cross(oh,or1);
        theta = np.arccos(np.dot(o2,o2new)/np.sqrt(np.sum(o2**2)*np.sum(o2new**2)))*180/3.1416;
        theta = theta * np.sign(np.dot(oh,np.cross(o2,o2new)));
        im_max = np.percentile(img.pixel_array.flatten(),99);
        res = imrotate(np.array(np.clip(np.array(img.pixel_array,dtype=np.float)/im_max*256,0,255),dtype=np.uint8),theta);

        if label is None:

            return res;
        else:
            return imrotate(label,theta);

    #Crop the image
    def crop_size(self, res):
        """
        Function that crops the image

        Args:
          res: numpy image array

        Returns:
          numpy array that has been cropped to 180 x 180

        """
        shift  = np.array([0,0])
        img_L=int(np.min(180)) #NEED TO UPDATE BASED ON COMMON IMAGE 

        if res.shape[0]>res.shape[1]:
            s = (res.shape[0]-res.shape[1])//2;
            res = res[s:s+res.shape[1],:];
            shift[1] = s;
        else:
            s = (res.shape[1]-res.shape[0])//2;
            res = res[:,s:s+res.shape[0]];
            shift[0] = s;

        #crop or stretch to the same size
        if img_L>0 and (res.shape[0] != img_L):
            #print("crop or fill",filename);
            if res.shape[0]>img_L:#because h=w after crop
                s = (res.shape[0]-img_L)//2;
                res = res[s:s+img_L,s:s+img_L];
                shift = shift + s;
            else:
                s = (img_L-res.shape[0])//2;
                res2 = np.zeros((img_L,img_L));
                res2[s:s+res.shape[0],s:s+res.shape[0]] = res;
                res = res2;
                shift = shift - s;

        return res

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

if __name__ == "__main__":
    """For testing purpose, use config to pass in the parms for Method2 class"""
    
    import config
    method = Method2(config)
    method.main_process()
