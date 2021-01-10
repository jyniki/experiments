'''
Author: Niki
Date: 2021-01-08 11:00:41
Description: 
'''
from training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from loss_functions.dice_loss import DC_and_topk_loss


class nnUNetTrainerV2_Loss_DiceTopK10(nnUNetTrainerV2):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.loss = DC_and_topk_loss({'batch_dice': self.batch_dice, 'smooth': 1e-5, 'do_bg': False},
                                     {'k': 10})
