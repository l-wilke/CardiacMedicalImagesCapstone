
"""Config file for running unet_predict"""

from keras.losses import binary_crossentropy


GPU_CLUSTER = "0,1"
per_process_gpu_memory_fraction = 0.4
model_name = "your_model_name"
image_size = 176 # or 256
weights_file = "/path/to/your/model/file"
image_file = "/masvol/heartsmart/unet_model/path1/data/combined_m_t_176_test_images.npy"
label_file = "/masvol/heartsmart/unet_model/path1/data/combined_m_t_176_test_labels.npy"
save_folder = "/path/to/save/results"
augmentation = True # or false
contrast_normalization = True # or False

'''
Sample command : 

python3 /masvol/heartsmart/Final/unet_model/unet_predict.py unet_predict_config

# old way
python3 /masvol/heartsmart/unet_model/unet_predict_sk.py pred_test 176 \
/masvol/heartsmart/unet_model/baseline/results/experiments/1_3_0_176_CLAHE_augx_bce1.hdf5 \
/masvol/output/dsb/norm/outlier_testing/m1t3/dsb_315_176_test.npy  none \
/masvol/heartsmart/unet_model/baseline/results/experiments/ False True   > runlogpred_test.txt &
'''
