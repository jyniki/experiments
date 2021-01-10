'''
Author: Niki
Date: 2020-12-30 18:03:03
Description: 
'''
from batchgenerators.utilities.file_and_folder_operations import *
from experiment_planning.DatasetAnalyzer import DatasetAnalyzer
from preprocessing.utils import crop
from paths import nnUNet_raw_data, nnUNet_cropped_data, preprocessing_output_dir
from configuration import default_num_threads
import shutil
from utils import convert_id_to_task_name
from preprocessing.sanity_checks import verify_dataset_integrity
from utils import recursive_find_python_class

def preprocess(task_ids,  pl3d= 'ExperimentPlanner3D_v21', pl2d = 'ExperimentPlanner2D_v21', no_pp = False, tl = default_num_threads, tf = default_num_threads, verify_integrity = False):
    '''
    param: 
        task_ids: List of integers belonging to the task ids you wish to run
        pl3d: Name of the ExperimentPlanner class for the full resolution 3D U-Net and U-Net cascade.
        pl2d: Name of the ExperimentPlanner class for the 2D U-Net
        no_pp: flag set true if you dont want to run the preprocessing. 
        tl: Number of processes used for preprocessing the low resolution data for the 3D low 
        tf: Number of processes used for preprocessing the full resolution data of the 2D U-Net and 3D U-Net
        verify_integrity: set this flag to check the dataset integrity
    '''
    dont_run_preprocessing = no_pp
    planner_name3d = pl3d
    planner_name2d = pl2d

    if planner_name3d == "None":
        planner_name3d = None
    if planner_name2d == "None":
        planner_name2d = None

    # we need raw data
    tasks = []
    for i in task_ids:
        i = int(i)

        task_name = convert_id_to_task_name(i)

        if verify_integrity:
            verify_dataset_integrity(join(nnUNet_raw_data, task_name))

        crop(task_name, False, tf)

        tasks.append(task_name)

    search_in = join(os.getcwd(),"experiment_planning")

    if planner_name3d is not None:
        planner_3d = recursive_find_python_class([search_in], planner_name3d, current_module="experiment_planning")
        if planner_3d is None:
            raise RuntimeError("Could not find the Planner class %s. Make sure it is located somewhere in experiment_planning" % planner_name3d)
    else:
        planner_3d = None

    if planner_name2d is not None:
        planner_2d = recursive_find_python_class([search_in], planner_name2d, current_module="experiment_planning")
        if planner_2d is None:
            raise RuntimeError("Could not find the Planner class %s. Make sure it is located somewhere in experiment_planning" % planner_name2d)
    else:
        planner_2d = None

    for t in tasks:
        print("\n\n\n", t)
        cropped_out_dir = os.path.join(nnUNet_cropped_data, t)
        preprocessing_output_dir_this_task = os.path.join(preprocessing_output_dir, t)
        
        dataset_json = load_json(join(cropped_out_dir, 'dataset.json'))
        modalities = list(dataset_json["modality"].values())
        collect_intensityproperties = True if (("CT" in modalities) or ("ct" in modalities)) else False
        dataset_analyzer = DatasetAnalyzer(cropped_out_dir, overwrite=False, num_processes=tf)  # this class creates the fingerprint
        _ = dataset_analyzer.analyze_dataset(collect_intensityproperties)  # this will write output files that will be used by the ExperimentPlanner


        maybe_mkdir_p(preprocessing_output_dir_this_task)
        shutil.copy(join(cropped_out_dir, "dataset_properties.pkl"), preprocessing_output_dir_this_task)
        shutil.copy(join(nnUNet_raw_data, t, "dataset.json"), preprocessing_output_dir_this_task)

        threads = (tl, tf)

        print("number of threads: ", threads, "\n")

        if planner_3d is not None:
            exp_planner = planner_3d(cropped_out_dir, preprocessing_output_dir_this_task)
            exp_planner.plan_experiment()
            if not dont_run_preprocessing:  # double negative, yooo
                exp_planner.run_preprocessing(threads)
        if planner_2d is not None:
            exp_planner = planner_2d(cropped_out_dir, preprocessing_output_dir_this_task)
            exp_planner.plan_experiment()
            if not dont_run_preprocessing:  # double negative, yooo
                exp_planner.run_preprocessing(threads)


