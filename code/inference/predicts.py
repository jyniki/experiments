import torch
from inference.predict_utils import predict_from_folder
from paths import DATASET_DIR, default_plans_identifier, network_training_output_dir, default_cascade_trainer, default_trainer
from batchgenerators.utilities.file_and_folder_operations import join, isdir
from utils import convert_id_to_task_name

def predict_simple(input_folder, output_folder, task_id, model, folds, save_npz, gpus, disable_mixed_precision):
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
    overwrite_existing = False
    mode = "normal"
    all_in_gpu = "None"
    task_name = task_id

    if not task_name.startswith("Task"):
        task_id = int(task_name)
        task_name = convert_id_to_task_name(task_id)

    assert model in ["2d", "3d_lowres", "3d_fullres", "3d_cascade_fullres"], "-m must be 2d, 3d_lowres, 3d_fullres or " \
                                                                             "3d_cascade_fullres"
    
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
        model_folder_name = join(network_training_output_dir, "3d_lowres", task_name, trainer_class_name + "__" +
                                  plans_identifier)
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

    model_folder_name = join(network_training_output_dir, model, task_name, trainer + "__" + plans_identifier)
    print("using model stored in ", model_folder_name)
    assert isdir(model_folder_name), "model output folder not found. Expected: %s" % model_folder_name

    predict_from_folder(model_folder_name, input_folder, output_folder, folds, save_npz, num_threads_preprocessing,
                        num_threads_nifti_save, lowres_segmentations, part_id, num_parts, not disable_tta,
                        overwrite_existing=overwrite_existing, mode=mode, overwrite_all_in_gpu=all_in_gpu,
                        mixed_precision=not disable_mixed_precision,
                        step_size=step_size, checkpoint_name="model_final_checkpoint")
# TODO
# def predict():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("-i", '--input_folder', help="Must contain all modalities for each patient in the correct"
#                                                      " order (same as training). Files must be named "
#                                                      "CASENAME_XXXX.nii.gz where XXXX is the modality "
#                                                      "identifier (0000, 0001, etc)", required=True)
#     parser.add_argument('-o', "--output_folder", required=True, help="folder for saving predictions")
#     parser.add_argument('-m', '--model_output_folder',
#                         help='model output folder. Will automatically discover the folds '
#                              'that were '
#                              'run and use those as an ensemble', required=True)
#     parser.add_argument('-f', '--folds', nargs='+', default='None', help="folds to use for prediction. Default is None "
#                                                                          "which means that folds will be detected "
#                                                                          "automatically in the model output folder")
#     parser.add_argument('-z', '--save_npz', required=False, action='store_true', help="use this if you want to ensemble"
#                                                                                       " these predictions with those of"
#                                                                                       " other models. Softmax "
#                                                                                       "probabilities will be saved as "
#                                                                                       "compresed numpy arrays in "
#                                                                                       "output_folder and can be merged "
#                                                                                       "between output_folders with "
#                                                                                       "merge_predictions.py")
#     parser.add_argument('-l', '--lowres_segmentations', required=False, default='None', help="if model is the highres "
#                                                                                              "stage of the cascade then you need to use -l to specify where the segmentations of the "
#                                                                                              "corresponding lowres unet are. Here they are required to do a prediction")
#     parser.add_argument("--part_id", type=int, required=False, default=0, help="Used to parallelize the prediction of "
#                                                                                "the folder over several GPUs. If you "
#                                                                                "want to use n GPUs to predict this "
#                                                                                "folder you need to run this command "
#                                                                                "n times with --part_id=0, ... n-1 and "
#                                                                                "--num_parts=n (each with a different "
#                                                                                "GPU (for example via "
#                                                                                "CUDA_VISIBLE_DEVICES=X)")
#     parser.add_argument("--num_parts", type=int, required=False, default=1,
#                         help="Used to parallelize the prediction of "
#                              "the folder over several GPUs. If you "
#                              "want to use n GPUs to predict this "
#                              "folder you need to run this command "
#                              "n times with --part_id=0, ... n-1 and "
#                              "--num_parts=n (each with a different "
#                              "GPU (via "
#                              "CUDA_VISIBLE_DEVICES=X)")
#     parser.add_argument("--num_threads_preprocessing", required=False, default=6, type=int, help=
#     "Determines many background processes will be used for data preprocessing. Reduce this if you "
#     "run into out of memory (RAM) problems. Default: 6")
#     parser.add_argument("--num_threads_nifti_save", required=False, default=2, type=int, help=
#     "Determines many background processes will be used for segmentation export. Reduce this if you "
#     "run into out of memory (RAM) problems. Default: 2")
#     parser.add_argument("--tta", required=False, type=int, default=1, help="Set to 0 to disable test time data "
#                                                                            "augmentation (speedup of factor "
#                                                                            "4(2D)/8(3D)), "
#                                                                            "lower quality segmentations")
#     parser.add_argument("--overwrite_existing", required=False, type=int, default=1, help="Set this to 0 if you need "
#                                                                                           "to resume a previous "
#                                                                                           "prediction. Default: 1 "
#                                                                                           "(=existing segmentations "
#                                                                                           "in output_folder will be "
#                                                                                           "overwritten)")
#     parser.add_argument("--mode", type=str, default="normal", required=False)
#     parser.add_argument("--all_in_gpu", type=str, default="None", required=False, help="can be None, False or True")
#     parser.add_argument("--step_size", type=float, default=0.5, required=False, help="don't touch")
#     # parser.add_argument("--interp_order", required=False, default=3, type=int,
#     #                     help="order of interpolation for segmentations, has no effect if mode=fastest")
#     # parser.add_argument("--interp_order_z", required=False, default=0, type=int,
#     #                     help="order of interpolation along z is z is done differently")
#     # parser.add_argument("--force_separate_z", required=False, default="None", type=str,
#     #                     help="force_separate_z resampling. Can be None, True or False, has no effect if mode=fastest")
#     parser.add_argument('--disable_mixed_precision', default=False, action='store_true', required=False,
#                         help='Predictions are done with mixed precision by default. This improves speed and reduces '
#                              'the required vram. If you want to disable mixed precision you can set this flag. Note '
#                              'that yhis is not recommended (mixed precision is ~2x faster!)')

#     args = parser.parse_args()
#     input_folder = args.input_folder
#     output_folder = args.output_folder
#     part_id = args.part_id
#     num_parts = args.num_parts
#     model = args.model_output_folder
#     folds = args.folds
#     save_npz = args.save_npz
#     lowres_segmentations = args.lowres_segmentations
#     num_threads_preprocessing = args.num_threads_preprocessing
#     num_threads_nifti_save = args.num_threads_nifti_save
#     tta = args.tta
#     step_size = args.step_size

#     # interp_order = args.interp_order
#     # interp_order_z = args.interp_order_z
#     # force_separate_z = args.force_separate_z

#     # if force_separate_z == "None":
#     #     force_separate_z = None
#     # elif force_separate_z == "False":
#     #     force_separate_z = False
#     # elif force_separate_z == "True":
#     #     force_separate_z = True
#     # else:
#     #     raise ValueError("force_separate_z must be None, True or False. Given: %s" % force_separate_z)

#     overwrite = args.overwrite_existing
#     mode = args.mode
#     all_in_gpu = args.all_in_gpu

#     if lowres_segmentations == "None":
#         lowres_segmentations = None

#     if isinstance(folds, list):
#         if folds[0] == 'all' and len(folds) == 1:
#             pass
#         else:
#             folds = [int(i) for i in folds]
#     elif folds == "None":
#         folds = None
#     else:
#         raise ValueError("Unexpected value for argument folds")

#     if tta == 0:
#         tta = False
#     elif tta == 1:
#         tta = True
#     else:
#         raise ValueError("Unexpected value for tta, Use 1 or 0")

#     if overwrite == 0:
#         overwrite = False
#     elif overwrite == 1:
#         overwrite = True
#     else:
#         raise ValueError("Unexpected value for overwrite, Use 1 or 0")

#     assert all_in_gpu in ['None', 'False', 'True']
#     if all_in_gpu == "None":
#         all_in_gpu = None
#     elif all_in_gpu == "True":
#         all_in_gpu = True
#     elif all_in_gpu == "False":
#         all_in_gpu = False

#     predict_from_folder(model, input_folder, output_folder, folds, save_npz, num_threads_preprocessing,
#                         num_threads_nifti_save, lowres_segmentations, part_id, num_parts, tta, mixed_precision=not args.disable_mixed_precision,
#                         overwrite_existing=overwrite, mode=mode, overwrite_all_in_gpu=all_in_gpu, step_size=step_size)
