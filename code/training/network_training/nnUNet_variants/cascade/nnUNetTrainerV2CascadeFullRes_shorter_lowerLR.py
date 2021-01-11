'''
Author: Niki
Date: 2021-01-09 23:52:21
Description: 
'''
from training.network_training.nnUNetTrainerV2_CascadeFullRes import nnUNetTrainerV2CascadeFullRes
class nnUNetTrainerV2CascadeFullRes_shorter_lowerLR(nnUNetTrainerV2CascadeFullRes):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, previous_trainer="nnUNetTrainerV2", fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory,
                         batch_dice, stage, unpack_data, deterministic,
                         previous_trainer, fp16)
        self.max_num_epochs = 500
        self.initial_lr = 1e-3
