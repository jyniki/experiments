'''
Author: Niki
Date: 2020-12-30 18:03:03
Description: 
'''
from batchgenerators.utilities.file_and_folder_operations import *
from run.default_configuration import get_default_configuration
from paths import default_plans_identifier
from training.cascade_stuff.predict_next_stage import predict_next_stage
from training.network_training.nnUNetTrainer import nnUNetTrainer
from training.network_training.nnUNetTrainerCascadeFullRes import nnUNetTrainerCascadeFullRes
from training.network_training.nnUNetTrainerV2_CascadeFullRes import nnUNetTrainerV2CascadeFullRes
from training.network_training.nnUNetTrainerV2_CascadeFullRes_DP import nnUNetTrainerV2CascadeFullRes_DP

from utils import convert_id_to_task_name

def train(network, network_trainer, task, fold, disable_saving, npz_flag, validation_only, continue_training, num_gpus, dbs_flag):
    plans_identifier = default_plans_identifier
    use_compressed_data = False
    decompress_data = not use_compressed_data
    deterministic = False
    valbest = False
    fp32 = False

    if not str(task).startswith("Task"):
        task_id = int(task)
        task = convert_id_to_task_name(task_id)

    if fold == 'all':
        pass
    else:
        fold = int(fold)

    print("number of GPUs:" + str(num_gpus))
    plans_file, output_folder_name, dataset_directory, batch_dice, stage, trainer_class = get_default_configuration(network, task, network_trainer, plans_identifier)

    if trainer_class is None:
        raise RuntimeError("Could not find trainer class")

    if network == "3d_cascade_fullres":
        assert issubclass(trainer_class, (nnUNetTrainerCascadeFullRes, nnUNetTrainerV2CascadeFullRes, nnUNetTrainerV2CascadeFullRes_DP)), \
            "If running 3d_cascade_fullres then your trainer class must be derived from nnUNetTrainerCascadeFullRes"
    else:
        assert issubclass(trainer_class, nnUNetTrainer), "network_trainer was found but is not derived from nnUNetTrainer"

    print(trainer_class)
    trainer = trainer_class(plans_file, fold, output_folder=output_folder_name,
                            dataset_directory=dataset_directory, batch_dice=batch_dice, stage=stage,
                            unpack_data=decompress_data, deterministic=deterministic,
                            distribute_batch_size=dbs_flag, num_gpus=num_gpus, fp16 = not fp32)

    if disable_saving:
        trainer.save_latest_only = False  # if false it will not store/overwrite _latest but separate files each
        trainer.save_intermediate_checkpoints = False  # whether or not to save checkpoint_latest
        trainer.save_best_checkpoint = False  # whether or not to save the best checkpoint according to self.best_val_eval_criterion_MA
        trainer.save_final_checkpoint = False  # whether or not to save the final checkpoint

    trainer.initialize(not validation_only)

    
    if not validation_only:
        if continue_training:
            trainer.load_latest_checkpoint()
        trainer.run_training()
    else:
        if valbest:
            trainer.load_best_checkpoint(train=False)
        else:
            trainer.load_latest_checkpoint(train=False)

    trainer.network.eval()


    trainer.data_init()
    
    if network == '3d_lowres':
        trainer.load_best_checkpoint(False)
        print("predicting segmentations for the next stage of the cascade")
        predict_next_stage(trainer, join(dataset_directory, trainer.plans['data_identifier'] + "_stage%d" % 1))

    # predict validation

    trainer.validate(save_softmax=npz_flag)