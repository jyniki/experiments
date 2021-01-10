'''
Author: Niki
Date: 2021-01-08 11:00:41
Description: 
'''
from training.loss_functions.deep_supervision import MultipleOutputLoss2
from training.loss_functions.dice_loss import DC_and_CE_loss
from training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2


class nnUNetTrainerV2_graduallyTransitionFromCEToDice(nnUNetTrainerV2):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.loss = DC_and_CE_loss({'batch_dice': self.batch_dice, 'smooth': 1e-5, 'do_bg': False}, {}, weight_ce=2, weight_dice=0)

    def update_loss(self):
        # we train the first 500 epochs with CE, then transition to Dice between 500 and 750. The last 250 epochs will be Dice only

        if self.epoch <= 500:
            weight_ce = 2
            weight_dice = 0
        elif 500 < self.epoch <= 750:
            weight_ce = 2 - 2 / 250 * (self.epoch - 500)
            weight_dice = 0 + 2 / 250 * (self.epoch - 500)
        elif 750 < self.epoch <= self.max_num_epochs:
            weight_ce = 0
            weight_dice = 2
        else:
            raise RuntimeError("Invalid epoch: %d" % self.epoch)

        self.print_to_log_file("weight ce", weight_ce, "weight dice", weight_dice)

        self.loss = DC_and_CE_loss({'batch_dice': self.batch_dice, 'smooth': 1e-5, 'do_bg': False}, {}, weight_ce=weight_ce,
                                   weight_dice=weight_dice)

        self.loss = MultipleOutputLoss2(self.loss, self.ds_loss_weights)

    def on_epoch_end(self):
        ret = super().on_epoch_end()
        self.update_loss()
        return ret

    def load_checkpoint_ram(self, checkpoint, train=True):
        ret = super().load_checkpoint_ram(checkpoint, train)
        self.update_loss()
        return ret
