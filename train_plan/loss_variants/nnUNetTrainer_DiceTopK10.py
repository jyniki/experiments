'''
Author: Niki
Date: 2021-01-12 00:16:59
Description: 
'''
from loss_functions.dice_loss import DC_and_topk_loss
# from nnunet.training.network_training import nnUNetTrainerCE
from training.network_training.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainer_DiceTopK10(nnUNetTrainer):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage,
                                              unpack_data, deterministic, fp16)
        k = 10
        self.loss = DC_and_topk_loss({'batch_dice':self.batch_dice, 'smooth':1e-5,
        	'do_bg':False}, {'k':k})
