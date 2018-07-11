
"""Config file for running unet_train"""

from keras.losses import binary_crossentropy

GPU_CLUSTER = "0,1"
per_process_gpu_memory_fraction = 0.4 # low so you can run other jobs in the same GPU device
model_name = "your_model_name"
image_size = 176 # or 256
training_images = "/masvol/heartsmart/unet_model/path1/data/combined_m_t_176_train_images.npy"
training_labels = "/masvol/heartsmart/unet_model/path1/data/combined_m_t_176_train_labels.npy"
test_images = "/masvol/heartsmart/unet_model/path1/data/combined_m_t_176_test_images.npy"
test_labels = "/masvol/heartsmart/unet_model/path1/data/combined_m_t_176_test_labels.npy"
model_path = "/masvol/heartsmar/unet_model/path"
dropout = True # or False
optimizer = "Adam"
lr = .00001 # learning rate
lossfn = binary_crossentropy
batch_size = 32 # or 16 or 64
epochs = 80 # or 100
augmentation = True
model_summary = False

''' 
 Example commandline arguments
 
 python3 /masvol/heartsmart/Final/unet_model/unet_train.py unet_train_config

 # old stuff
 python3 /masvol/heartsmart/unet_model/unet_train_sk.py 1_3_0_176_CLAHE_augx_drop_bce1 176 \
/masvol/output/acdc/norm/1/3/unet_model/data16bit/combined_1_3_176_train_images.npy \
/masvol/output/acdc/norm/1/3/unet_model/data16bit/combined_1_3_176_train_labels.npy \
/masvol/output/acdc/norm/1/3/unet_model/data16bit/combined_1_3_176_test_images.npy \
/masvol/output/acdc/norm/1/3/unet_model/data16bit/combined_1_3_176_test_labels.npy \
/masvol/heartsmart/unet_model/baseline/results/experiments/ True Adam .00005 binary_crossentropy \
4 100 True True  > runlog_1_3_0_176_CLAHE_augx_drop_bce1.txt &

python3 /masvol/heartsmart/unet_model/unet_train_sk.py 1_3_0_176_CLAHE_augx_bce1 176 \
/masvol/output/acdc/norm/1/3/unet_model/data16bit/combined_1_3_176_train_images.npy \
/masvol/output/acdc/norm/1/3/unet_model/data16bit/combined_1_3_176_train_labels.npy \
/masvol/output/acdc/norm/1/3/unet_model/data16bit/combined_1_3_176_test_images.npy \
/masvol/output/acdc/norm/1/3/unet_model/data16bit/combined_1_3_176_test_labels.npy \
/masvol/heartsmart/unet_model/baseline/results/experiments/ False Adam .00005 binary_crossentropy \
8 100 True True  > runlog_1_3_0_176_CLAHE_augx_bce1.txt &
'''
