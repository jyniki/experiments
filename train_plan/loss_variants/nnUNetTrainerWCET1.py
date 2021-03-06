'''
Author: Niki
Date: 2021-01-12 00:17:01
Description: 
'''
from loss_functions.ND_Crossentropy import WeightedCrossEntropyLoss
from training.network_training.nnUNetTrainer import nnUNetTrainer
import torch


class nnUNetTrainerWCET1(nnUNetTrainer):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super(nnUNetTrainerWCET1, self).__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage,
                                              unpack_data, deterministic, fp16)
        self.weight = torch.cuda.FloatTensor([1.0,576.4]) # pre-defined according to the task
        self.loss = WeightedCrossEntropyLoss(self.weight)
