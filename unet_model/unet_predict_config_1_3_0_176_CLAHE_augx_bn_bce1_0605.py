
"""Config file for running unet_predict"""


GPU_CLUSTER = "4,7"
GPUs = len(GPU_CLUSTER.split(','))
per_process_gpu_memory_fraction = 0.2
model_name = '1_3_0_176_CLAHE_augx_bn_bce1' #combined_1_3_0_176_CLAHE_augx_bce1_crERDS2
image_size = 176 # or 256-
source = 'test'
# source = 'train'
# source = 'validate'
source_type = 'test'
# source_type = 'train'
# source_type = 'validate'
method = 1
Type = 3
hdf5_path = "/masvol/heartsmart/unet_model/baseline/results/final/{0}.hdf5".format(model_name)
predict_path = "combined_{0}_crERDS2605".format(model_name)
volume_path = "{0}{1}".format(Type,predict_path)
augmentation = False #True # or False
contrast_normalization = True # or True
batch_normalization = True # or False
file_source = "dsb"
dropout = False
patient_list = []


'''
Sample command : 

python3 /masvol/heartsmart/Final/unet_model/unet_predict.py unet_predict_config

# old way
python3 /masvol/heartsmart/unet_model/unet_predict_sk.py pred_test 176 \
/masvol/heartsmart/unet_model/baseline/results/experiments/1_3_0_176_CLAHE_augx_bce1.hdf5 \
/masvol/output/dsb/norm/outlier_testing/m1t3/dsb_315_176_test.npy  none \
/masvol/heartsmart/unet_model/baseline/results/experiments/ False True   > runlogpred_test.txt &
'''

