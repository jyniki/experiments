'''
Author: Niki
Date: 2021-01-08 11:00:41
Description: 
'''
from loss_functions.crossentropy import RobustCrossEntropyLoss
from training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2


class nnUNetTrainerV2_Loss_CE(nnUNetTrainerV2):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.loss = RobustCrossEntropyLoss()
