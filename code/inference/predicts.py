'''
Author: Niki
Date: 2021-01-10 13:59:06
Description: 
'''
import torch
from inference.predict_utils import predict_from_folder
from paths import DATASET_DIR, default_plans_identifier, preprocessing_output_dir, \
    network_training_output_dir, pre_training_output_dir, default_cascade_trainer, default_trainer

from batchgenerators.utilities.file_and_folder_operations import join, isdir, os
from utils import convert_id_to_task_name
from evaluation.evaluator import aggregate_scores
from configuration import default_num_threads


def predict_simple(input_folder, output_folder, task_id, 
                   model, folds, save_npz, gpus, disable_mixed_precision, 
                   mode, using_pretrain, overwrite_existing, eval_flag):
    # default_trainer: nnUNetTrainerV2, can change to nnUNetTrainerV2_DP or nnUNetTrainerV2_DDP
    trainer_class_name = default_trainer 
    cascade_trainer_class_name = default_cascade_trainer 
    plans_identifier = default_plans_identifier
    num_threads_preprocessing = 6
    num_threads_nifti_save = 2
    
    input_folder = join(DATASET_DIR,input_folder)
    part_id = gpus-1
    num_parts = gpus
    disable_tta = False
    step_size = 0.5
    all_in_gpu = "None"
    task_name = task_id

    if not task_name.startswith("Task"):
        task_id = int(task_name)
        task_name = convert_id_to_task_name(task_id)

    assert model in ["2d", "3d_lowres", "3d_fullres", "3d_cascade_fullres"], "-m must be 2d, 3d_lowres, 3d_fullres or 3d_cascade_fullres"
    
    output_folder = join(DATASET_DIR,output_folder,model,task_name)

    lowres_segmentations = None

    if isinstance(folds, list):
        if folds[0] == 'all' and len(folds) == 1:
            pass
        else:
            folds = [int(i) for i in folds]
    elif folds == "None":
        folds = None
    else:
        raise ValueError("Unexpected value for argument folds")

    assert all_in_gpu in ['None', 'False', 'True']
    if all_in_gpu == "None":
        all_in_gpu = None
    elif all_in_gpu == "True":
        all_in_gpu = True
    elif all_in_gpu == "False":
        all_in_gpu = False

    if model == "3d_cascade_fullres" and lowres_segmentations is None:
        print("lowres_segmentations is None. Attempting to predict 3d_lowres first...")
        assert part_id == 0 and num_parts == 1, "if you don't specify a --lowres_segmentations folder for the " \
                                                "inference of the cascade, custom values for part_id and num_parts " \
                                                "are not supported. If you wish to have multiple parts, please " \
                                                "run the 3d_lowres inference first (separately)"
        if using_pretrain:
            model_folder_name = join(pre_training_output_dir, "3d_lowres", task_name, trainer_class_name + "__" + plans_identifier)
        else:
            model_folder_name = join(network_training_output_dir, "3d_lowres", task_name, trainer_class_name + "__" + plans_identifier)
        assert isdir(model_folder_name), "model output folder not found. Expected: %s" % model_folder_name
        lowres_output_folder = join(output_folder, "3d_lowres_predictions")
        
        predict_from_folder(model_folder_name, input_folder, lowres_output_folder, folds, False,
                            num_threads_preprocessing, num_threads_nifti_save, None, part_id, num_parts, not disable_tta,
                            overwrite_existing=overwrite_existing, mode=mode, overwrite_all_in_gpu=all_in_gpu,
                            mixed_precision=not disable_mixed_precision,
                            step_size=step_size)
        lowres_segmentations = lowres_output_folder
        torch.cuda.empty_cache()
        print("3d_lowres done")

    if model == "3d_cascade_fullres":
        trainer = cascade_trainer_class_name
    else:
        trainer = trainer_class_name
    
    if using_pretrain:
        model_folder_name = join(pre_training_output_dir, model, task_name, trainer + "__" + plans_identifier)
    else:
        model_folder_name = join(network_training_output_dir, model, task_name, trainer + "__" + plans_identifier)
    print("using model stored in ", model_folder_name)
    assert isdir(model_folder_name), "model output folder not found. Expected: %s" % model_folder_name

    predict_from_folder(model_folder_name, input_folder, output_folder, folds, save_npz, num_threads_preprocessing,
                        num_threads_nifti_save, lowres_segmentations, part_id, num_parts, not disable_tta,
                        overwrite_existing=overwrite_existing, mode=mode, overwrite_all_in_gpu=all_in_gpu,
                        mixed_precision=not disable_mixed_precision,
                        step_size=step_size, checkpoint_name="model_final_checkpoint")

    # TODO
    if eval_flag:
        task = output_folder.split('/')[-1]
        gt_folder = join(preprocessing_output_dir,task,"gt_segmentations")
        predict_val(output_folder,gt_folder)
    
def predict_val(pre_folder,gt_folder):
    import pickle  
    pred_gt_tuples = []
    for fname in os.listdir(pre_folder):
        if(fname.split('.')[-1]=="gz"):
            pred_gt_tuples.append([join(pre_folder, fname), join(gt_folder, fname)])
    task = pre_folder.split('/')[-2]
    
    f = open(join(pre_folder,"plans.pkl"),'rb')  
    info = pickle.load(f)  
    num_classes = info['num_classes'] + 1  # background is no longer in num_classes
    _ = aggregate_scores(pred_gt_tuples, labels=list(range(num_classes)),
                         json_output_file=join(pre_folder, "summary.json"),
                         json_task=task, num_threads=default_num_threads)
        
