# CardiacMedicalImagesCapstone

Capstone Project for Data Science & Engineering Master's Program at University of California, San Diego (UCSD)

In the diagnosis of the heart disease, one of the parameters that cardiologist examine is the volume ejected by the left ventricle.The difference of the end-diastolic volume (EDV) and end-systolic volume (ESV), which is a measure of the amount of blood that is pumped in one cardiac cycle is a parameter that is used in the diagnosis of the process. From the volume, the ejection fraction (EF) can be derived which is the ratio of (EDV - ESV)/EDV.

MRI images enable the ability to estimate the ESV, EDV, and EF. Currently, it takes imaging technicians and doctors several minutes to read the images to come to a diagnosis and it is not an easily repeatable process. Automation of parts of the process to determine cardiac parameters and/ or function can lead to faster consistent diagnosis and create a repeatable process for diagnosis.

Over the past several years, LV segmentation algorithms have been evolved but they have had limited success due to the lack of available labeled data. Over the years, more datasets have been made available publically which has resulted in improvement in LV segmentation algorithms and more methods have been developed to help with the problem.

We have experimented with many of these proposed approaches and compared their effectiveness on publicly available datasets. Our work offers an analytics pipeline consisting of the best methods we found for preprocessing, modeling, and postprocessing cardiac MRI images for LV segmentation and volume estimation using deep learning.

Our approach segments the LV cavity in each short-axis (SAX) slice, computes the LV area and volume for each slice, then sums all per-slice LV volumes to estimate the overall LV volume. We use a U-Net, a deep learning model originally created for image segmentation in biomedical applications but have been applied to other domains as well. In our approach, a U-Net is trained on the segmentation task by predicting the LV contour in each input CMR image. The contours are then used as input to a separate process to calculate the LV volume. In this volume calculation process, ES and ED frames are determined in each slice, then summed across all slices to determine the LV volume for each patient.

##  Data
Over the past few years, there has been an increasing number of publicly available cardiac MRI Images with labels that identify the contours of the Left Ventricle and other cardiac features. In 2009, the Sunnybrook Cardiac Data (SCD) was made available through the Cardiac MR Left Ventricle Segmentation Challenge. Kaggle.com organized the Second Annual Data Science Bowl (DSB) in 2015 which provided the largest set of MRI images. In 2017, the Medical Image Computing & Computer Assisted Intervention (MIACCI) organized the Automated Cardiac Diagnosis Challenge (ACDC) that made an additional set of MRI heart images publically available. All three datasets were utilized for this effort.

There are three different types of views that an MRI Image can take in order to examine the heart. These views are the 4-Chamber view, 2-Chamber view, and the Short Axis view. For this analysis, the focus was on using the Short Axis (SAX) views. For the SAX view, a singular heart is broken up into multiple slices, each slice is the part of the heart at a different physical location. Each slice consists of multiple frames that show the heart across one cardiac cycle, temporal aspect. 

The SCD data consists of 45-cine MRI images (1.6 GB) from a mixed group of patients and pathologies: healthy, hypertrophy, heart failure with infarction, and heart failure without infraction. The MRI images are in the DICOM image format that consists of several metadata parameters about the patient and the image. For each patient record, there is a set of hand drawn contours. The contours were drawn by Perry Radau from the Sunnybrook Health Science Centre. The contours were available in text files that consisted of the contour points, which needed to be converted into an image. 

The ACDC dataset consists of SAX MRI Images for 100 patients (3.3 GB) in the NIFTI image format. Similar to the DICOM images, the NIFTI images consist of metadata about the patient and the image. Each patient directory consists of a 4-D NIFTI Format Images. Contour files have been provided for the End-Systolic and End-Diastolic images for each patient. These contours were drawn to follow the limit defined by the aortic valve. This method for defining how the contours were drawn may differ from how the SCD contours were drawn as they were drawnby a different individual. 

The third dataset is the DSB dataset which consists of MRI images for 1,140 patients (100GB). The dataset includes other views of the heart, 4-Chamber view, and 2-Chamber view, in addition to the SAX view. This dataset did not come with contour labels but rather provides an End-Systolic Volume (ESV) and End-Diastolic Volume (EDV) for each patient. These volumes are derived from the contours that are drawn from the MRI images. Dr. Anai’s group from the National Heart Lung and Blood Institute drew the contours for the DSB data, which were the basis of the ESV and EDV calculations. The group’s methodology for identifying the LV contours differs from the groups that drew the contours for the SCD and ACDC datasets. Dr. Anai’s group looks to where one can see left ventricular muscle in order to draw the contour, resulting in a “half-moon” contour or a partial slice instead of a circular contour. 

##  Authors

1. Ehab Abdelmaguid 
2. Jolene Huang 
3. Sanjay Kenchareddy 
4. Disha Singla 
5. Laura Wilke

##  Acknowledgements

Thank you to our advisors Mai Nguyen and Ilkay Altintas.

Thank you to everyone who provided their expertise and assisted on this project:
1. Marcus Bobar
2. Eric Carruth
3. Dr. Evan Muse
4. Dylan Uys
5. Gary Cottrell



