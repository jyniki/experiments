from copy import deepcopy
from typing import Tuple, Union, List
import numpy as np
from batchgenerators.augmentations.utils import resize_segmentation
from inference.segmentation_export import save_segmentation_nifti_from_softmax, save_segmentation_nifti
from batchgenerators.utilities.file_and_folder_operations import *
from multiprocessing import Process, Queue
import torch
import SimpleITK as sitk
import shutil
from multiprocessing import Pool
from preprocessing.connected_components import load_remove_save, load_postprocessing
from training.model_restore import load_model_and_checkpoint_files
from training.network_training.nnUNetTrainer import nnUNetTrainer
from utils import to_one_hot


def preprocess_save_to_queue(preprocess_fn, q, list_of_lists, output_files, segs_from_prev_stage, classes, transpose_forward):

    errors_in = []
    for i, l in enumerate(list_of_lists):
        try:
            output_file = output_files[i]
            print("preprocessing", output_file)
            d, _, dct = preprocess_fn(l)
            print(output_file, dct)
            if segs_from_prev_stage[i] is not None:
                assert isfile(segs_from_prev_stage[i]) and segs_from_prev_stage[i].endswith(
                    ".nii.gz"), "segs_from_prev_stage must point to a segmentation file"
                seg_prev = sitk.GetArrayFromImage(sitk.ReadImage(segs_from_prev_stage[i]))
                # check to see if shapes match
                img = sitk.GetArrayFromImage(sitk.ReadImage(l[0]))
                assert all([i == j for i, j in zip(seg_prev.shape, img.shape)]), "image and segmentation from previous " \
                                                                                 "stage don't have the same pixel array " \
                                                                                 "shape! image: %s, seg_prev: %s" % \
                                                                                 (l[0], segs_from_prev_stage[i])
                seg_prev = seg_prev.transpose(transpose_forward)
                seg_reshaped = resize_segmentation(seg_prev, d.shape[1:], order=1, cval=0)
                seg_reshaped = to_one_hot(seg_reshaped, classes)
                d = np.vstack((d, seg_reshaped)).astype(np.float32)
            print(d.shape)
            if np.prod(d.shape) > (2e9 / 4 * 0.85):  # *0.85 just to be save, 4 because float32 is 4 bytes
                print(
                    "This output is too large for python process-process communication. "
                    "Saving output temporarily to disk")
                np.save(output_file[:-7] + ".npy", d)
                d = output_file[:-7] + ".npy"
            q.put((output_file, (d, dct)))
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except Exception as e:
            print("error in", l)
            print(e)
    q.put("end")
    if len(errors_in) > 0:
        print("There were some errors in the following cases:", errors_in)
        print("These cases were ignored.")
    else:
        print("This worker has ended successfully, no errors to report")
    # restore output
    # sys.stdout = sys.__stdout__


def preprocess_multithreaded(trainer, list_of_lists, output_files, num_processes=2, segs_from_prev_stage=None):
    if segs_from_prev_stage is None:
        segs_from_prev_stage = [None] * len(list_of_lists)

    num_processes = min(len(list_of_lists), num_processes)

    classes = list(range(1, trainer.num_classes))
    assert isinstance(trainer, nnUNetTrainer)
    q = Queue(1)
    processes = []
    for i in range(num_processes):
        pr = Process(target=preprocess_save_to_queue, args=(trainer.preprocess_patient, q,
                                                            list_of_lists[i::num_processes],
                                                            output_files[i::num_processes],
                                                            segs_from_prev_stage[i::num_processes],
                                                            classes, trainer.plans['transpose_forward']))
        pr.start()
        processes.append(pr)

    try:
        end_ctr = 0
        while end_ctr != num_processes:
            item = q.get()
            if item == "end":
                end_ctr += 1
                continue
            else:
                yield item

    finally:
        for p in processes:
            if p.is_alive():
                p.terminate()
            p.join()

        q.close()


def predict_cases(model, list_of_lists, output_filenames, folds, save_npz, num_threads_preprocessing,
                  num_threads_nifti_save, segs_from_prev_stage=None, do_tta=True, mixed_precision=True, overwrite_existing=False,
                  all_in_gpu=False, step_size=0.5, checkpoint_name="model_final_checkpoint",
                  segmentation_export_kwargs: dict = None):

    assert len(list_of_lists) == len(output_filenames)
    if segs_from_prev_stage is not None: assert len(segs_from_prev_stage) == len(output_filenames)

    pool = Pool(num_threads_nifti_save)
    results = []

    cleaned_output_files = []
    for o in output_filenames:
        dr, f = os.path.split(o)
        if len(dr) > 0:
            maybe_mkdir_p(dr)
        if not f.endswith(".nii.gz"):
            f, _ = os.path.splitext(f)
            f = f + ".nii.gz"
        cleaned_output_files.append(join(dr, f))

    if not overwrite_existing:
        print("number of cases:", len(list_of_lists))
        not_done_idx = [i for i, j in enumerate(cleaned_output_files) if not isfile(j)]
        cleaned_output_files = [cleaned_output_files[i] for i in not_done_idx]
        list_of_lists = [list_of_lists[i] for i in not_done_idx]
        if segs_from_prev_stage is not None:
            segs_from_prev_stage = [segs_from_prev_stage[i] for i in not_done_idx]

        print("number of cases that still need to be predicted:", len(cleaned_output_files))

    print("emptying cuda cache")
    torch.cuda.empty_cache()

    print("loading parameters for folds,", folds)
    trainer, params = load_model_and_checkpoint_files(model, folds, mixed_precision=mixed_precision, checkpoint_name=checkpoint_name)

    if segmentation_export_kwargs is None:
        if 'segmentation_export_params' in trainer.plans.keys():
            force_separate_z = trainer.plans['segmentation_export_params']['force_separate_z']
            interpolation_order = trainer.plans['segmentation_export_params']['interpolation_order']
            interpolation_order_z = trainer.plans['segmentation_export_params']['interpolation_order_z']
        else:
            force_separate_z = None
            interpolation_order = 1
            interpolation_order_z = 0
    else:
        force_separate_z = segmentation_export_kwargs['force_separate_z']
        interpolation_order = segmentation_export_kwargs['interpolation_order']
        interpolation_order_z = segmentation_export_kwargs['interpolation_order_z']

    print("starting preprocessing generator")
    preprocessing = preprocess_multithreaded(trainer, list_of_lists, cleaned_output_files, num_threads_preprocessing, segs_from_prev_stage)
    print("starting prediction...")
    all_output_files = []
    for preprocessed in preprocessing:
        output_filename, (d, dct) = preprocessed
        all_output_files.append(all_output_files)
        if isinstance(d, str):
            data = np.load(d)
            os.remove(d)
            d = data

        print("predicting", output_filename)
        softmax = []
        for p in params:
            trainer.load_checkpoint_ram(p, False)
            softmax.append(trainer.predict_preprocessed_data_return_seg_and_softmax(
                d, do_mirroring=do_tta, mirror_axes=trainer.data_aug_params['mirror_axes'], use_sliding_window=True,
                step_size=step_size, use_gaussian=True, all_in_gpu=all_in_gpu,
                mixed_precision=mixed_precision)[1][None])

        softmax = np.vstack(softmax)
        softmax_mean = np.mean(softmax, 0)

        transpose_forward = trainer.plans.get('transpose_forward')
        if transpose_forward is not None:
            transpose_backward = trainer.plans.get('transpose_backward')
            softmax_mean = softmax_mean.transpose([0] + [i + 1 for i in transpose_backward])

        if save_npz:
            npz_file = output_filename[:-7] + ".npz"
        else:
            npz_file = None

        if hasattr(trainer, 'regions_class_order'):
            region_class_order = trainer.regions_class_order
        else:
            region_class_order = None

        bytes_per_voxel = 4
        if all_in_gpu:
            bytes_per_voxel = 2  # if all_in_gpu then the return value is half (float16)
        if np.prod(softmax_mean.shape) > (2e9 / bytes_per_voxel * 0.85):  # * 0.85 just to be save
            print( "This output is too large for python process-process communication. Saving output temporarily to disk")
            np.save(output_filename[:-7] + ".npy", softmax_mean)
            softmax_mean = output_filename[:-7] + ".npy"

        results.append(pool.starmap_async(save_segmentation_nifti_from_softmax,
                                          ((softmax_mean, output_filename, dct, interpolation_order, region_class_order,
                                            None, None,npz_file, None, force_separate_z, interpolation_order_z),)))
        

    print("inference done. Now waiting for the segmentation export to finish...")
    _ = [i.get() for i in results]
    results = []
    pp_file = join(model, "postprocessing.json")
    if isfile(pp_file):
        print("postprocessing...")
        shutil.copy(pp_file, os.path.abspath(os.path.dirname(output_filenames[0])))
        for_which_classes, min_valid_obj_size = load_postprocessing(pp_file)
        results.append(pool.starmap_async(load_remove_save,
                                          zip(output_filenames, output_filenames,
                                              [for_which_classes] * len(output_filenames),
                                              [min_valid_obj_size] * len(output_filenames))))
        _ = [i.get() for i in results]
    else:
        print("WARNING! Cannot run postprocessing because the postprocessing file is missing. Make sure to run "
              "consolidate_folds in the output folder of the model first!\nThe folder you need to run this in is "
              "%s" % model)

    pool.close()
    pool.join()


def predict_cases_fast(model, list_of_lists, output_filenames, folds, num_threads_preprocessing,
                       num_threads_nifti_save, segs_from_prev_stage=None, do_tta=True, mixed_precision=True,
                       overwrite_existing=False, all_in_gpu=False, step_size=0.5, checkpoint_name="model_final_checkpoint",
                       segmentation_export_kwargs: dict = None):
    assert len(list_of_lists) == len(output_filenames)
    if segs_from_prev_stage is not None: assert len(segs_from_prev_stage) == len(output_filenames)

    pool = Pool(num_threads_nifti_save)
    results = []

    cleaned_output_files = []
    for o in output_filenames:
        dr, f = os.path.split(o)
        if len(dr) > 0:
            maybe_mkdir_p(dr)
        if not f.endswith(".nii.gz"):
            f, _ = os.path.splitext(f)
            f = f + ".nii.gz"
        cleaned_output_files.append(join(dr, f))

    if not overwrite_existing:
        print("number of cases:", len(list_of_lists))
        not_done_idx = [i for i, j in enumerate(cleaned_output_files) if not isfile(j)]

        cleaned_output_files = [cleaned_output_files[i] for i in not_done_idx]
        list_of_lists = [list_of_lists[i] for i in not_done_idx]
        if segs_from_prev_stage is not None:
            segs_from_prev_stage = [segs_from_prev_stage[i] for i in not_done_idx]

        print("number of cases that still need to be predicted:", len(cleaned_output_files))

    print("emptying cuda cache")
    torch.cuda.empty_cache()

    print("loading parameters for folds,", folds)
    trainer, params = load_model_and_checkpoint_files(model, folds, mixed_precision=mixed_precision, checkpoint_name=checkpoint_name)

    if segmentation_export_kwargs is None:
        if 'segmentation_export_params' in trainer.plans.keys():
            force_separate_z = trainer.plans['segmentation_export_params']['force_separate_z']
            interpolation_order = trainer.plans['segmentation_export_params']['interpolation_order']
            interpolation_order_z = trainer.plans['segmentation_export_params']['interpolation_order_z']
        else:
            force_separate_z = None
            interpolation_order = 1
            interpolation_order_z = 0
    else:
        force_separate_z = segmentation_export_kwargs['force_separate_z']
        interpolation_order = segmentation_export_kwargs['interpolation_order']
        interpolation_order_z = segmentation_export_kwargs['interpolation_order_z']

    print("starting preprocessing generator")
    preprocessing = preprocess_multithreaded(trainer, list_of_lists, cleaned_output_files, num_threads_preprocessing,
                                             segs_from_prev_stage)

    print("starting prediction...")
    for preprocessed in preprocessing:
        print("getting data from preprocessor")
        output_filename, (d, dct) = preprocessed
        print("got something")
        if isinstance(d, str):
            print("what I got is a string, so I need to load a file")
            data = np.load(d)
            os.remove(d)
            d = data

        # preallocate the output arrays
        # same dtype as the return value in predict_preprocessed_data_return_seg_and_softmax (saves time)
        softmax_aggr = None  # np.zeros((trainer.num_classes, *d.shape[1:]), dtype=np.float16)
        all_seg_outputs = np.zeros((len(params), *d.shape[1:]), dtype=int)
        print("predicting", output_filename)

        for i, p in enumerate(params):
            trainer.load_checkpoint_ram(p, False)

            res = trainer.predict_preprocessed_data_return_seg_and_softmax(d, do_mirroring=do_tta,
                                                                           mirror_axes=trainer.data_aug_params['mirror_axes'],
                                                                           use_sliding_window=True,
                                                                           step_size=step_size, use_gaussian=True,
                                                                           all_in_gpu=all_in_gpu,
                                                                           mixed_precision=mixed_precision)

            if len(params) > 1:
                # otherwise we dont need this and we can save ourselves the time it takes to copy that
                print("aggregating softmax")
                if softmax_aggr is None:
                    softmax_aggr = res[1]
                else:
                    softmax_aggr += res[1]
            all_seg_outputs[i] = res[0]

        print("obtaining segmentation map")
        if len(params) > 1:
            # we dont need to normalize the softmax by 1 / len(params) because this would not change the outcome of the argmax
            seg = softmax_aggr.argmax(0)
        else:
            seg = all_seg_outputs[0]

        print("applying transpose_backward")
        transpose_forward = trainer.plans.get('transpose_forward')
        if transpose_forward is not None:
            transpose_backward = trainer.plans.get('transpose_backward')
            seg = seg.transpose([i for i in transpose_backward])

        print("initializing segmentation export")
        results.append(pool.starmap_async(save_segmentation_nifti,
                                          ((seg, output_filename, dct, interpolation_order, force_separate_z,
                                            interpolation_order_z),)))
        print("done")

    print("inference done. Now waiting for the segmentation export to finish...")
    _ = [i.get() for i in results]
    # now apply postprocessing
    # first load the postprocessing properties if they are present. Else raise a well visible warning
    results = []
    pp_file = join(model, "postprocessing.json")
    if isfile(pp_file):
        print("postprocessing...")
        shutil.copy(pp_file, os.path.dirname(output_filenames[0]))
        # for_which_classes stores for which of the classes everything but the largest connected component needs to be
        # removed
        for_which_classes, min_valid_obj_size = load_postprocessing(pp_file)
        results.append(pool.starmap_async(load_remove_save,
                                          zip(output_filenames, output_filenames,
                                              [for_which_classes] * len(output_filenames),
                                              [min_valid_obj_size] * len(output_filenames))))
        _ = [i.get() for i in results]
    else:
        print("WARNING! Cannot run postprocessing because the postprocessing file is missing. Make sure to run "
              "consolidate_folds in the output folder of the model first!\nThe folder you need to run this in is "
              "%s" % model)

    pool.close()
    pool.join()


def predict_cases_fastest(model, list_of_lists, output_filenames, folds, num_threads_preprocessing,
                          num_threads_nifti_save, segs_from_prev_stage=None, do_tta=True, mixed_precision=True,
                          overwrite_existing=False, all_in_gpu=True, step_size=0.5,
                          checkpoint_name="model_final_checkpoint"):
    assert len(list_of_lists) == len(output_filenames)
    if segs_from_prev_stage is not None: assert len(segs_from_prev_stage) == len(output_filenames)

    pool = Pool(num_threads_nifti_save)
    results = []

    cleaned_output_files = []
    for o in output_filenames:
        dr, f = os.path.split(o)
        if len(dr) > 0:
            maybe_mkdir_p(dr)
        if not f.endswith(".nii.gz"):
            f, _ = os.path.splitext(f)
            f = f + ".nii.gz"
        cleaned_output_files.append(join(dr, f))

    if not overwrite_existing:
        print("number of cases:", len(list_of_lists))
        not_done_idx = [i for i, j in enumerate(cleaned_output_files) if not isfile(j)]

        cleaned_output_files = [cleaned_output_files[i] for i in not_done_idx]
        list_of_lists = [list_of_lists[i] for i in not_done_idx]
        if segs_from_prev_stage is not None:
            segs_from_prev_stage = [segs_from_prev_stage[i] for i in not_done_idx]

        print("number of cases that still need to be predicted:", len(cleaned_output_files))

    print("emptying cuda cache")
    torch.cuda.empty_cache()

    print("loading parameters for folds,", folds)
    trainer, params = load_model_and_checkpoint_files(model, folds, mixed_precision=mixed_precision, checkpoint_name=checkpoint_name)

    print("starting preprocessing generator")
    preprocessing = preprocess_multithreaded(trainer, list_of_lists, cleaned_output_files, num_threads_preprocessing,
                                             segs_from_prev_stage)

    print("starting prediction...")
    for preprocessed in preprocessing:
        print("getting data from preprocessor")
        output_filename, (d, dct) = preprocessed
        print("got something")
        if isinstance(d, str):
            print("what I got is a string, so I need to load a file")
            data = np.load(d)
            os.remove(d)
            d = data

        # preallocate the output arrays
        # same dtype as the return value in predict_preprocessed_data_return_seg_and_softmax (saves time)
        all_softmax_outputs = np.zeros((len(params), trainer.num_classes, *d.shape[1:]), dtype=np.float16)
        all_seg_outputs = np.zeros((len(params), *d.shape[1:]), dtype=int)
        print("predicting", output_filename)

        for i, p in enumerate(params):
            trainer.load_checkpoint_ram(p, False)
            res = trainer.predict_preprocessed_data_return_seg_and_softmax(d, do_mirroring=do_tta,
                                                                           mirror_axes=trainer.data_aug_params['mirror_axes'],
                                                                           use_sliding_window=True,
                                                                           step_size=step_size, use_gaussian=True,
                                                                           all_in_gpu=all_in_gpu,
                                                                           mixed_precision=mixed_precision)
            if len(params) > 1:
                # otherwise we dont need this and we can save ourselves the time it takes to copy that
                all_softmax_outputs[i] = res[1]
            all_seg_outputs[i] = res[0]

        print("aggregating predictions")
        if len(params) > 1:
            softmax_mean = np.mean(all_softmax_outputs, 0)
            seg = softmax_mean.argmax(0)
        else:
            seg = all_seg_outputs[0]

        print("applying transpose_backward")
        transpose_forward = trainer.plans.get('transpose_forward')
        if transpose_forward is not None:
            transpose_backward = trainer.plans.get('transpose_backward')
            seg = seg.transpose([i for i in transpose_backward])

        print("initializing segmentation export")
        results.append(pool.starmap_async(save_segmentation_nifti,((seg, output_filename, dct, 0, None),)))
        print("done")

    print("inference done. Now waiting for the segmentation export to finish...")
    _ = [i.get() for i in results]
    # now apply postprocessing
    # first load the postprocessing properties if they are present. Else raise a well visible warning
    results = []
    pp_file = join(model, "postprocessing.json")
    if isfile(pp_file):
        print("postprocessing...")
        shutil.copy(pp_file, os.path.dirname(output_filenames[0]))
        # for_which_classes stores for which of the classes everything but the largest connected component needs to be
        # removed
        for_which_classes, min_valid_obj_size = load_postprocessing(pp_file)
        results.append(pool.starmap_async(load_remove_save,zip(output_filenames, output_filenames,[for_which_classes] * len(output_filenames),
                                              [min_valid_obj_size] * len(output_filenames))))
        _ = [i.get() for i in results]
    else:
        print("WARNING! Cannot run postprocessing because the postprocessing file is missing. Make sure to run "
              "consolidate_folds in the output folder of the model first!\nThe folder you need to run this in is "
              "%s" % model)

    pool.close()
    pool.join()


def check_input_folder_and_return_caseIDs(input_folder, expected_num_modalities):
    print("This model expects %d input modalities for each image" % expected_num_modalities)
    files = subfiles(input_folder, suffix=".nii.gz", join=False, sort=True)

    maybe_case_ids = np.unique([i[:-12] for i in files])

    remaining = deepcopy(files)
    missing = []

    assert len(files) > 0, "input folder did not contain any images (expected to find .nii.gz file endings)"

    # now check if all required files are present and that no unexpected files are remaining
    for c in maybe_case_ids:
        for n in range(expected_num_modalities):
            expected_output_file = c + "_%04.0d.nii.gz" % n
            if not isfile(join(input_folder, expected_output_file)):
                missing.append(expected_output_file)
            else:
                remaining.remove(expected_output_file)

    print("Found %d unique case ids, here are some examples:" % len(maybe_case_ids),
          np.random.choice(maybe_case_ids, min(len(maybe_case_ids), 10)))
    print("If they don't look right, make sure to double check your filenames. They must end with _0000.nii.gz etc")

    if len(remaining) > 0:
        print("found %d unexpected remaining files in the folder. Here are some examples:" % len(remaining),
              np.random.choice(remaining, min(len(remaining), 10)))

    if len(missing) > 0:
        print("Some files are missing:")
        print(missing)
        raise RuntimeError("missing files in input_folder")

    return maybe_case_ids


def predict_from_folder(model: str, input_folder: str, output_folder: str, folds: Union[Tuple[int], List[int]],
                        save_npz: bool, num_threads_preprocessing: int, num_threads_nifti_save: int,
                        lowres_segmentations: Union[str, None],
                        part_id: int, num_parts: int, tta: bool, mixed_precision: bool = True,
                        overwrite_existing: bool = True, mode: str = 'normal', overwrite_all_in_gpu: bool = None,
                        step_size: float = 0.5, checkpoint_name: str = "model_final_checkpoint",
                        segmentation_export_kwargs: dict = None):
    
    maybe_mkdir_p(output_folder)
    shutil.copy(join(model, 'plans.pkl'), output_folder)

    assert isfile(join(model, "plans.pkl")), "Folder with saved model weights must contain a plans.pkl file"
    expected_num_modalities = load_pickle(join(model, "plans.pkl"))['num_modalities']

    # check input folder integrity
    case_ids = check_input_folder_and_return_caseIDs(input_folder, expected_num_modalities)

    output_files = [join(output_folder, i + ".nii.gz") for i in case_ids]
    all_files = subfiles(input_folder, suffix=".nii.gz", join=False, sort=True)
    list_of_lists = [[join(input_folder, i) for i in all_files if i[:len(j)].startswith(j) and len(i) == (len(j) + 12)] for j in case_ids]

    if lowres_segmentations is not None:
        assert isdir(lowres_segmentations), "if lowres_segmentations is not None then it must point to a directory"
        lowres_segmentations = [join(lowres_segmentations, i + ".nii.gz") for i in case_ids]
        assert all([isfile(i) for i in lowres_segmentations]), "not all lowres_segmentations files are present. " \
                                                               "(I was searching for case_id.nii.gz in that folder)"
        lowres_segmentations = lowres_segmentations[part_id::num_parts]
    else:
        lowres_segmentations = None

    if mode == "normal":
        if overwrite_all_in_gpu is None:
            all_in_gpu = False
        else:
            all_in_gpu = overwrite_all_in_gpu

        return predict_cases(model, list_of_lists[part_id::num_parts], output_files[part_id::num_parts], folds,
                             save_npz, num_threads_preprocessing, num_threads_nifti_save, lowres_segmentations, tta,
                             mixed_precision=mixed_precision, overwrite_existing=overwrite_existing, all_in_gpu=all_in_gpu,
                             step_size=step_size, checkpoint_name=checkpoint_name,
                             segmentation_export_kwargs=segmentation_export_kwargs)
    elif mode == "fast":
        if overwrite_all_in_gpu is None:
            all_in_gpu = True
        else:
            all_in_gpu = overwrite_all_in_gpu

        # assert save_npz is False
        save_npz = False
        return predict_cases_fast(model, list_of_lists[part_id::num_parts], output_files[part_id::num_parts], folds,
                                  num_threads_preprocessing, num_threads_nifti_save, lowres_segmentations,
                                  tta, mixed_precision=mixed_precision, overwrite_existing=overwrite_existing, all_in_gpu=all_in_gpu,
                                  step_size=step_size, checkpoint_name=checkpoint_name,
                                  segmentation_export_kwargs=segmentation_export_kwargs)
    elif mode == "fastest":
        if overwrite_all_in_gpu is None:
            all_in_gpu = True
        else:
            all_in_gpu = overwrite_all_in_gpu

        # assert save_npz is False
        save_npz = False
        return predict_cases_fastest(model, list_of_lists[part_id::num_parts], output_files[part_id::num_parts], folds,
                                     num_threads_preprocessing, num_threads_nifti_save, lowres_segmentations,
                                     tta, mixed_precision=mixed_precision, overwrite_existing=overwrite_existing, all_in_gpu=all_in_gpu,
                                     step_size=step_size, checkpoint_name=checkpoint_name)
    else:
        raise ValueError("unrecognized mode. Must be normal, fast or fastest")

