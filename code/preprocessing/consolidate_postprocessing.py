'''
Author: Niki
Date: 2021-01-08 10:51:48
Description: 
'''
import shutil
from typing import Tuple
from batchgenerators.utilities.file_and_folder_operations import *
from configuration import default_num_threads
from evaluation.evaluator import aggregate_scores
from preprocessing.connected_components import determine_postprocessing

def collect_cv_niftis(cv_folder: str, output_folder: str, validation_folder_name: str = 'validation_raw',
                      folds: tuple = (0, 1, 2, 3, 4)):
    folders_folds = [join(cv_folder, "fold_%d" % i) for i in folds]

    assert all([isdir(i) for i in folders_folds]), "some folds are missing"

    # now for each fold, read the postprocessing json. this will tell us what the name of the validation folder is
    validation_raw_folders = [join(cv_folder, "fold_%d" % i, validation_folder_name) for i in folds]

    # now copy all raw niftis into cv_niftis_raw
    maybe_mkdir_p(output_folder)
    for f in folds:
        niftis = subfiles(validation_raw_folders[f], suffix=".nii.gz")
        for n in niftis:
            shutil.copy(n, join(output_folder))


def consolidate_folds(output_folder_base, validation_folder_name: str = 'validation_raw',
                      advanced_postprocessing: bool = False, folds: Tuple[int] = (0, 1, 2, 3, 4)):
   
    output_folder_raw = join(output_folder_base, "cv_niftis_raw")
    output_folder_gt = join(output_folder_base, "gt_niftis")
    collect_cv_niftis(output_folder_base, output_folder_raw, validation_folder_name,
                      folds)

    num_niftis_gt = len(subfiles(join(output_folder_base, "gt_niftis")))
    # count niftis in there
    num_niftis = len(subfiles(output_folder_raw))
    if num_niftis != num_niftis_gt:
        shutil.rmtree(output_folder_raw)
        raise AssertionError("If does not seem like you trained all the folds! Train all folds first!")

    # load a summary file so that we can know what class labels to expect
    summary_fold0 = load_json(join(output_folder_base, "fold_0", validation_folder_name, "summary.json"))['results'][
        'mean']
    classes = [int(i) for i in summary_fold0.keys()]
    niftis = subfiles(output_folder_raw, join=False, suffix=".nii.gz")
    test_pred_pairs = [(join(output_folder_gt, i), join(output_folder_raw, i)) for i in niftis]

    # determine_postprocessing needs a summary.json file in the folder where the raw predictions are. We could compute
    # that from the summary files of the five folds but I am feeling lazy today
    aggregate_scores(test_pred_pairs, labels=classes, json_output_file=join(output_folder_raw, "summary.json"),
                     num_threads=default_num_threads)

    determine_postprocessing(output_folder_base, output_folder_gt, 'cv_niftis_raw',
                             final_subf_name="cv_niftis_postprocessed", processes=default_num_threads,
                             advanced_postprocessing=advanced_postprocessing)
    