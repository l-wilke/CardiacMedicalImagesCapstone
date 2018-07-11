
For training a model, please take a look at the unet_train_config.py file
Make a copy of it and edit your copy to set the variables 

Feed unet_train_config file into unet_train.py to train a model

For example

python3 unet_train.py unet_train_config_your_copy


For predicting, please make a copy of any of the unet_predict* files and issue a command like the following

python3 unet_predict.py unet_predicting_config_your_version


unet_predict.py includes removing extra contours from the predictions and minimum and maximum of the LV contours in each slice.


If ensamble of the predictions is needed, please use post_processing.py


Locate one of the post_processing_config* files and issue a command like the following

python3 post_processing.py post_processing_config_your_version





The following four variables are important to assign when running in mutli-gpu mode

os.environ["CUDA_VISIBLE_DEVICES"] = GPU_CLUSTER
GPU_CLUSTER is for specifying the GPU devices.  

GPUs is set to the number of GPU devices specified.

config.gpu_options.allow_growth = True

allow_growth variable is set to True so that the memory is not allocated to the maximum when your script starts.

config.gpu_options.per_process_gpu_memory_fraction = 0.4

per_process_gpu_memory_fraction is set to adjust how much memory to allocate on each GPU device.  


Note: issue nvidia-smi to see which device is available.  The middle column from nvidia-smi view shows the memory usage

