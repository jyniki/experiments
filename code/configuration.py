'''
Author: Niki
Date: 2020-12-30 18:03:03
Description: 
'''
import yaml
config = yaml.load(open('./configs.yaml', 'r'), Loader=yaml.FullLoader)

default_num_threads = config["nnUNet_def_n_proc"]
if config['preprocessing_args']['tl']:
    tl = config['preprocessing_args']['tl']
else:
    tl = default_num_threads
  
if config['preprocessing_args']['tf']:
    tf = config['preprocessing_args']['tf']
else:
    tf = default_num_threads
    
    
# determines what threshold to use for resampling the low resolution axis separately (with NN)
RESAMPLING_SEPARATE_Z_ANISO_THRESHOLD = 3  