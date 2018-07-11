#!/usr/bin/env python


import os
import glob

#source = "validate"
#patient = 597
#patient = 234
#patient = 123

#/masvol/data/dsb/train/234/study/sax_19/IM-3097-0008-0001.dcm
#['', 'masvol', 'data', 'dsb', 'train', '234', 'study', 'sax_19', 'IM-3097-0008-0001.dcm']

def copying_files(patient, source):
    """ This Module copies the misplaced dcm files from one sax folder to another
    

    Args:
      patient:  patient folder id
      source:  train, validate, or test

    Returns:  not returning, but only copying files from the wrong sax folder to the new sax folder

    """
    mainpath = "/masvol/data/dsb/{0}/{1}/study".format(source, patient)
    origpath = "{0}/sax*/*".format(mainpath)

    count = 0

    for i in glob.glob(origpath):
       if not i.endswith('.dcm'):
           continue

       print (i) # original file
       nodes = i.split('/')
       print (nodes)
       filename = nodes[-1]
       print (filename) # original filename without path
       filenodes = filename.split('-')
       if len(filenodes) != 4:
           continue

       sax = filenodes[-1].replace('.dcm','')

       newdir = "{0}/sax_{1}".format(mainpath, int(sax))
       print (newdir) # new sax folder
       newname = newdir + '/' + '-'.join(filenodes[:-1]) + '.dcm'
       print (newname) # new dcm filename

       newdirpath = os.path.dirname(newname)

       if not os.path.exists(newdirpath):
           os.makedirs(newdirpath)
       
       #os.rename(i, newname)
       os.popen("cp {0} {1}".format(i, newname)) # copying original from old sax to the new sax folder
       count += 1

       #if count > 5:
       #    break

if __name__ == "__main__":
    patient = 499
    source = "train"
    copying_files(patient, source)
