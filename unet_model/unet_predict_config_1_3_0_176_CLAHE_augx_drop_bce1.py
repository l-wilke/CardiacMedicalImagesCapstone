
"""Config file for running unet_predict"""


GPU_CLUSTER = "4,5"
GPUs = len(GPU_CLUSTER.split(','))
per_process_gpu_memory_fraction = 0.4
model_name = '1_3_0_176_CLAHE_augx_drop_bce1'
image_size = 176 # or 256
source = 'test'
# source = 'validate'
source_type = 'test'
# source_type = 'validate'
method = 1
Type = 3
hdf5_path = "/masvol/heartsmart/unet_model/baseline/results/final/{0}.hdf5".format(model_name)
predict_path = "combined_{0}_crERDS2".format(model_name)
volume_path = "{0}{1}".format(Type,predict_path)
augmentation = False #True or False, False for now
contrast_normalization = True # or True
batch_normalization = False # or False
file_source = "dsb"
dropout = True
patient_list = []

