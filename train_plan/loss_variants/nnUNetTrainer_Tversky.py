'''
Author: Niki
Date: 2021-01-12 00:17:01
Description: 
'''
from loss_functions.dice_loss import TverskyLoss
from training.network_training.nnUNetTrainer import nnUNetTrainer
from utils import softmax_helper


class nnUNetTrainer_Tversky(nnUNetTrainer):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage,
                                              unpack_data, deterministic, fp16)
        self.apply_nonlin = softmax_helper
        self.loss = TverskyLoss(apply_nonlin=self.apply_nonlin, batch_dice=self.batch_dice, smooth=1e-5, do_bg=False)
