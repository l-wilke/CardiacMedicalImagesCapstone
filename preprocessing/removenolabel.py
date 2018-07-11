
import os
import glob

#imgpath = "/opt/output/sunnybrook/norm/1/3/images/*"
#imgpath = "/opt/output/sunnybrook/norm/2/1/images/*"
#imgpath = "/masvol/output/sunnybrook/norm/1/1/images/*"

def match_and_keep(imgpath, labelpath):
    """
    We only want to keep the dicom numpy array files which has a label file

    Args:
      imgpath: The directory name where the dicom numpy array files are
      labelpath: The directory name of the labels

    Returns: Nothing returned, but the dicom numpy array files without any label files will be deleted

    """

    for i in glob.glob(imgpath):
        print (i)
        found = 0
        notfound = 0

        for j in glob.glob("{0}/*".format(i)):
            #labelfile = j.replace('images','challenge')
            labelfile = j.replace('images','labels')
            labelfile = labelfile.replace('dcm','dcm.label')

            if os.path.isfile(labelfile):
                found += 1
                continue

            print (j)
            os.remove(j)
            notfound += 1

        print (found, notfound)

if __name__ == "__main__":
    method = 1 # 1 or 2
    normtype = 1 # 1, 2, or 3
    labelpath = "labels" # or challenge, directory path where the labels are

    imgpath = "/masvol/output/sunnybrook/norm/{0}/{1}/images/*".format(method, normtype)
    match_and_keep(imagepath, labelpath)
