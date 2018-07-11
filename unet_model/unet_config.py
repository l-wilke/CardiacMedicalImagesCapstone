
"""Config file for running unet_train"""

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
lossfn = binary_crossentropy # or log_dice_loss, etc.
batch_size = 32 # or 16 or 64
epochs = 80 # or 100
augmentation = True
model_summary = False
