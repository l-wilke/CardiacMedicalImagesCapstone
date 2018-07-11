
input_dir = '/masvol/data/dsb'
base_dir = '/masvol/output/dsb/norm/1/3/unet_model_'
output_dir = '/masvol/output/dsb/norm/1/3/ensamble/'
systolic_path = 'top_test_systolic_vote'
diastolic_path = 'top_test_diastolic_vote'
sources = ['test']
volume_dir = '/masvol/output/dsb/volume/1/3top_test_vote'
systolic_models = []
diastolic_models = ['combined_1_3_0_176_CLAHE_augx_bce1_crERDS2',
                    'combined_1_3_0_176_CLAHE_augx_drop_bce1_crERDS2',
                    'combined_1_3_0_176_CLAHE_augx_bn_bce1_crERDS2']
#sources = ['train','validate']



ensamble_type = 1 # 1 for vote and 0 for average
image_size = 176
