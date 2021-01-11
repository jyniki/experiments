'''
Author: Niki
Date: 2020-12-30 18:03:03
Description: 
'''
import yaml
config = yaml.load(open('./configs/default.yaml', 'r'), Loader=yaml.FullLoader)

default_num_threads = config["nnUNet_def_n_proc"]
if config['tl']:
    tl = config['tl']
else:
    tl = default_num_threads
  
if config['tf']:
    tf = config['tf']
else:
    tf = default_num_threads
    
default_trainer=  config['default_trainer']         
default_cascade_trainer = config['default_cascade_trainer']  
  
# determines what threshold to use for resampling the low resolution axis separately (with NN)
RESAMPLING_SEPARATE_Z_ANISO_THRESHOLD = 3  

network_config = yaml.load(open('./configs/network_trainer.yaml', 'r'), Loader=yaml.FullLoader)
MAX_EPOCH = config['max_num_epochs']

