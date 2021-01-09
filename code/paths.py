'''
Author: Niki
Date: 2020-12-30 18:03:02
Description: change to local
'''

import yaml   
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p, join

config = yaml.load(open('./configs/default.yaml', 'r'), Loader=yaml.FullLoader)
default_plans_identifier = config['default_plans_identifier']
default_data_identifier = config['default_data_identifier']
default_trainer = config['default_trainer']
default_cascade_trainer = config['default_cascade_trainer'] 

DATASET_DIR = config['DATASET_DIR']
my_output_identifier = config['output_identifier']
pretrain_identifier = config['pretrain_identifier']
base = join(DATASET_DIR,"nnUNet_raw") if DATASET_DIR else None
preprocessing_output_dir = join(DATASET_DIR, "nnUNet_preprocessed") if DATASET_DIR else None
network_training_output_dir_base = join(DATASET_DIR, "nnUNet_trained_models") if DATASET_DIR else None

if base is not None:
    maybe_mkdir_p(base)
    nnUNet_raw_data = join(base, "nnUNet_raw_data")
    nnUNet_cropped_data = join(base, "nnUNet_cropped_data")
    maybe_mkdir_p(nnUNet_raw_data)
    maybe_mkdir_p(nnUNet_cropped_data)
else:
    print("the path of nnUNet_raw_data_base is not defined, please check configs.yaml.")
    nnUNet_cropped_data = nnUNet_raw_data = None

if preprocessing_output_dir is not None:
    maybe_mkdir_p(preprocessing_output_dir)
else:
    print("the path of nnUNet_preprocessed is not defined and nnU-Net can not be used for preprocessing or training."
          "Please check configs.yaml.")
    preprocessing_output_dir = None

if network_training_output_dir_base is not None:
    network_training_output_dir = join(network_training_output_dir_base, my_output_identifier)
    pre_training_output_dir = join(network_training_output_dir_base, pretrain_identifier)
    maybe_mkdir_p(network_training_output_dir)
else:
    print("RESULTS_FOLDER is not defined and nnU-Net cannot be used for training or inference."
          "Please check configs.yaml.")
    network_training_output_dir = None
