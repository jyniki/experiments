'''
Author: Niki
Date: 2021-01-07 13:52:47
Description: 
'''

from configuration import default_num_threads, tf, tl
from paths import nnUNet_raw_data
from batchgenerators.utilities.file_and_folder_operations import join
import yaml
import argparse
import os

def pipeline():
    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", type=str, default='./configs/configs.yaml')
    parser.add_argument("-o", "--operation", type=int, nargs="+", default=0)
    
    args = parser.parse_args()
    operation_type = args.operation
    config = yaml.load(open(args.yaml, 'r'), Loader=yaml.FullLoader)
    if 0 in operation_type:
        print( "operation_type: \n"
                "   1 -- convert_decathlon \n"
                "   2 -- data_preprocess \n"
                "   3 -- training_parse \n"
                "   4 -- inference_parse \n"
                "   5 -- download_pretrained_model \n")
        
    if 1 in operation_type:
        print("convert_decathlon")
        from preprocessing import nnUNet_convert_decathlon_task as cdt
        original_data = join(nnUNet_raw_data,config['convert_decathlon']['original_data'])
        cdt.convert_decathlon(original_data, default_num_threads)

    if 2 in operation_type:
        from preprocessing import nnUNet_plan_and_preprocess as preprocess
        task_ids = config['preprocessing_args']['task_ids']
        pl3d = config['preprocessing_args']['pl3d']
        pl2d = config['preprocessing_args']['pl2d']
        no_pp = config['preprocessing_args']['no_pp']
        verify_integrity = config['preprocessing_args']['verify_integrity']
        preprocess.preprocess(task_ids = task_ids, pl3d = pl3d, pl2d = pl2d, no_pp = no_pp,tl = tl, tf = tf, verify_integrity = verify_integrity)
    
    if 3 in operation_type:
        from run import run_training, run_training_DP, run_training_DDP
        task = config['training_args']['task_id']
        fold = config['training_args']['fold']
        network = config['training_args']['network']
        validation_only = config['training_args']['validation_only']
        disable_saving = config['training_args']['disable_saving']
        continue_training = config['training_args']['continue_training']
        npz_flag = config['training_args']['npz']
        
        if config['train_module'] == 'dp':
            print("training_dp")
            network_trainer = config['training_dp']['network_trainer']
            gpus = config['training_dp']['gpus']
            dbs_flag = config['training_dp']['dbs']
            run_training_DP.train(network,network_trainer,task,fold,disable_saving, npz_flag, validation_only,continue_training, gpus, dbs_flag)
        
        elif config['train_module'] == 'ddp':
            print("training_ddp")
            gpus = config['training_ddp']['gpus']
            network_trainer = config['training_ddp']['network_trainer']
            device_id = config['training_ddp']['device_id']
            command = "/data0/JY/anaconda3/envs/myenv/bin/python -m torch.distributed.launch --master_port=4321 --nproc_per_node=%d run/run_training_DDP.py %s %s %d %d "%(gpus,network,network_trainer,task,fold)
            dbs_flag = config['training_ddp']['dbs']
            
            # python -m torch.distributed.launch --master_port=4321 --nproc_per_node=4 run/run_training_DDP.py 3d_fullres nnUNetTrainerV2_DDP 8 4 --local_rank 1
            print(command)
            if not dbs_flag:
                os.system(command)
            else:
                os.system(command+"--dbs")
            
        else:
            print("training_single_gpu")
            network_trainer = config['training_single']['network_trainer']
            run_training.train(network,network_trainer,task,fold,disable_saving, npz_flag, validation_only,continue_training)
            
    if 4 in operation_type:
        print("inference_parse")
        from inference import predicts
        input_folder = config['inference_args']['input_folder']
        output_folder = config['inference_args']['output_folder']
        task_id = str(config['inference_args']['task_id'])
        model = config['inference_args']['model']
        folds = config['inference_args']['folds']
        save_npz = config['inference_args']['save_npz']
        gpus = config['inference_args']['gpus']
        disable_mixed_precision = config['inference_args']['disable_mixed_precision']
        predicts.predict_simple(input_folder, output_folder, task_id, model, folds, save_npz, gpus, disable_mixed_precision)
        
    
if __name__ == "__main__":
    pipeline()