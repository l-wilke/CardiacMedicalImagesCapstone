
"""Config file for running unet_predict"""


GPU_CLUSTER = "4,5,6,7"
#GPUs = len(GPU_CLUSTER.split(','))
GPUs = 0
per_process_gpu_memory_fraction = 0.3
model_name = '1_3_0_176_aug_drop_dice' #combined_1_3_0_176_CLAHE_augx_bce1_crERDS2
image_size = 176 # or 256
source = 'test'
# source = 'validate'
#source = 'train'
source_type = 'test'
# source_type = 'validate'
# source_type = 'train'
method = 1
Type = 3
hdf5_path = "/masvol/heartsmart/unet_model/baseline/results/final/{0}.hdf5".format(model_name)
predict_path = "combined_{0}_crERDS2".format(model_name)
volume_path = "{0}{1}".format(Type,predict_path)
augmentation = False #True # or False
contrast_normalization = False # or True
batch_normalization = False # or False
file_source = "dsb"
dropout = True
patient_list = []

