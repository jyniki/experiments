from multiprocessing.pool import Pool
from time import sleep
import matplotlib
from configuration import default_num_threads
from preprocessing.connected_components import determine_postprocessing
from training.data_augmentation.default_data_augmentation import get_moreDA_augmentation
from training.dataloading.dataset_loading import DataLoader3D, unpack_dataset
from evaluation.evaluator import aggregate_scores
from network_architecture.neural_network import SegmentationNetwork
from paths import network_training_output_dir
from inference.segmentation_export import save_segmentation_nifti_from_softmax
from batchgenerators.utilities.file_and_folder_operations import *
import numpy as np
from training.network_training.nnUNetTrainerV2_DP import nnUNetTrainerV2_DP
from torch.nn.parallel.data_parallel import DataParallel
from utils import to_one_hot
import shutil
from torch import nn
matplotlib.use("agg")

class nnUNetTrainerV2CascadeFullRes_DP(nnUNetTrainerV2_DP):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, previous_trainer="nnUNetTrainerV2_DP", num_gpus=1, 
                 distribute_batch_size=False, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, 
                          unpack_data, deterministic, fp16, num_gpus, distribute_batch_size)
        self.init_args = (plans_file, fold, output_folder, dataset_directory, batch_dice, stage,
                          unpack_data, deterministic, fp16, num_gpus, distribute_batch_size, previous_trainer)

        self.num_gpus = num_gpus
        self.distribute_batch_size = distribute_batch_size
        
        if self.output_folder is not None:
            task = self.output_folder.split("/")[-3]
            plans_identifier = self.output_folder.split("/")[-2].split("__")[-1]

            folder_with_segs_prev_stage = join(network_training_output_dir, "3d_lowres", task, previous_trainer + "__" + plans_identifier, "pred_next_stage")
            self.folder_with_segs_from_prev_stage = folder_with_segs_prev_stage
        else:
            self.folder_with_segs_from_prev_stage = None
        print(self.folder_with_segs_from_prev_stage)

    def do_split(self):
        super().do_split()
        for k in self.dataset:
            self.dataset[k]['seg_from_prev_stage_file'] = join(self.folder_with_segs_from_prev_stage, k + "_segFromPrevStage.npz")
            assert isfile(self.dataset[k]['seg_from_prev_stage_file']), \
                "seg from prev stage missing: %s" % (self.dataset[k]['seg_from_prev_stage_file'])
        for k in self.dataset_val:
            self.dataset_val[k]['seg_from_prev_stage_file'] = join(self.folder_with_segs_from_prev_stage, k + "_segFromPrevStage.npz")
        for k in self.dataset_tr:
            self.dataset_tr[k]['seg_from_prev_stage_file'] = join(self.folder_with_segs_from_prev_stage, k + "_segFromPrevStage.npz")

    def get_basic_generators(self):
        self.load_dataset()
        self.do_split()
        if self.threeD:
            dl_tr = DataLoader3D(self.dataset_tr, self.basic_generator_patch_size, self.patch_size, self.batch_size,
                                 True, oversample_foreground_percent=self.oversample_foreground_percent,
                                 pad_mode="constant", pad_sides=self.pad_all_sides)
            dl_val = DataLoader3D(self.dataset_val, self.patch_size, self.patch_size, self.batch_size, True,
                                  oversample_foreground_percent=self.oversample_foreground_percent,
                                  pad_mode="constant", pad_sides=self.pad_all_sides)
        else:
            raise NotImplementedError("2D has no cascade")

        return dl_tr, dl_val

    def process_plans(self, plans):
        super().process_plans(plans)
        self.num_input_channels += (self.num_classes - 1)  # for seg from prev stage

    def setup_DA_params(self):
        super().setup_DA_params()
        self.data_aug_params["num_cached_per_thread"] = 2
        self.data_aug_params['move_last_seg_chanel_to_data'] = True
        self.data_aug_params['cascade_do_cascade_augmentations'] = True
        self.data_aug_params['cascade_random_binary_transform_p'] = 0.4
        self.data_aug_params['cascade_random_binary_transform_p_per_label'] = 1
        self.data_aug_params['cascade_random_binary_transform_size'] = (1, 8)
        self.data_aug_params['cascade_remove_conn_comp_p'] = 0.2
        self.data_aug_params['cascade_remove_conn_comp_max_size_percent_threshold'] = 0.15
        self.data_aug_params['cascade_remove_conn_comp_fill_with_other_class_p'] = 0.0
        self.data_aug_params['selected_seg_channels'] = [0, 1]
        self.data_aug_params['all_segmentation_labels'] = list(range(1, self.num_classes))

    def initialize(self, training=True, force_load_plans=False):

        if not self.was_initialized:
            if force_load_plans or (self.plans is None):
                self.load_plans_file()

            self.process_plans(self.plans)
            self.setup_DA_params()

            ################# Here we wrap the loss for deep supervision ############
            net_numpool = len(self.net_num_pool_op_kernel_sizes)
            weights = np.array([1 / (2 ** i) for i in range(net_numpool)])
            mask = np.array([True if i < net_numpool - 1 else False for i in range(net_numpool)])
            weights[~mask] = 0
            weights = weights / weights.sum()
            self.ds_loss_weights = weights
            ################# END ###################

            self.folder_with_preprocessed_data = join(self.dataset_directory, self.plans['data_identifier'] + "_stage%d" % self.stage)

            if training:
                if not isdir(self.folder_with_segs_from_prev_stage):
                    raise RuntimeError(
                        "Cannot run final stage of cascade. Run corresponding 3d_lowres first and predict the "
                        "segmentations for the next stage")

                self.dl_tr, self.dl_val = self.get_basic_generators()
                if self.unpack_data:
                    print("unpacking dataset")
                    unpack_dataset(self.folder_with_preprocessed_data)
                    print("done")
                else:
                    print(
                        "INFO: Not unpacking data! Training may be slow due to that. Pray you are not using 2d or you "
                        "will wait all winter for your model to finish!")

                self.tr_gen, self.val_gen = get_moreDA_augmentation(self.dl_tr, self.dl_val,
                                                                    self.data_aug_params['patch_size_for_spatialtransform'],
                                                                    self.data_aug_params,
                                                                    deep_supervision_scales=self.deep_supervision_scales,
                                                                    pin_memory=self.pin_memory)
                self.print_to_log_file("TRAINING KEYS:\n %s" % (str(self.dataset_tr.keys())), also_print_to_console=False)
                self.print_to_log_file("VALIDATION KEYS:\n %s" % (str(self.dataset_val.keys())), also_print_to_console=False)
            else:
                pass

            self.initialize_network()
            self.initialize_optimizer_and_scheduler()

            assert isinstance(self.network, (SegmentationNetwork, DataParallel))
        else:
            self.print_to_log_file('self.was_initialized is True, not running self.initialize again')

        self.was_initialized = True

    def validate(self, do_mirroring: bool = True, use_sliding_window: bool = True, step_size: float = 0.5,
                 save_softmax: bool = True, use_gaussian: bool = True, overwrite: bool = True,
                 validation_folder_name: str = 'validation_raw', debug: bool = False, all_in_gpu: bool = False,
                 segmentation_export_kwargs: dict = None):
        assert self.was_initialized, "must initialize, ideally with checkpoint (or train first)"

        current_mode = self.network.training
        self.network.eval()
        # save whether network is in deep supervision mode or not
        ds = self.network.do_ds
        # disable deep supervision
        self.network.do_ds = False

        if segmentation_export_kwargs is None:
            if 'segmentation_export_params' in self.plans.keys():
                force_separate_z = self.plans['segmentation_export_params']['force_separate_z']
                interpolation_order = self.plans['segmentation_export_params']['interpolation_order']
                interpolation_order_z = self.plans['segmentation_export_params']['interpolation_order_z']
            else:
                force_separate_z = None
                interpolation_order = 1
                interpolation_order_z = 0
        else:
            force_separate_z = segmentation_export_kwargs['force_separate_z']
            interpolation_order = segmentation_export_kwargs['interpolation_order']
            interpolation_order_z = segmentation_export_kwargs['interpolation_order_z']

        if self.dataset_val is None:
            self.load_dataset()
            self.do_split()

        output_folder = join(self.output_folder, validation_folder_name)
        maybe_mkdir_p(output_folder)
        # this is for debug purposes
        my_input_args = {'do_mirroring': do_mirroring,
                         'use_sliding_window': use_sliding_window,
                         'step': step_size,
                         'save_softmax': save_softmax,
                         'use_gaussian': use_gaussian,
                         'overwrite': overwrite,
                         'validation_folder_name': validation_folder_name,
                         'debug': debug,
                         'all_in_gpu': all_in_gpu,
                         'segmentation_export_kwargs': segmentation_export_kwargs,
                         }
        save_json(my_input_args, join(output_folder, "validation_args.json"))

        if do_mirroring:
            if not self.data_aug_params['do_mirror']:
                raise RuntimeError("We did not train with mirroring so you cannot do inference with mirroring enabled")
            mirror_axes = self.data_aug_params['mirror_axes']
        else:
            mirror_axes = ()

        pred_gt_tuples = []

        export_pool = Pool(default_num_threads)
        results = []

        for k in self.dataset_val.keys():
            properties = load_pickle(self.dataset[k]['properties_file'])
            fname = properties['list_of_data_files'][0].split("/")[-1][:-12]

            if overwrite or (not isfile(join(output_folder, fname + ".nii.gz"))) or \
                    (save_softmax and not isfile(join(output_folder, fname + ".npz"))):
                data = np.load(self.dataset[k]['data_file'])['data']

                # concat segmentation of previous step
                seg_from_prev_stage = np.load(join(self.folder_with_segs_from_prev_stage,
                                                   k + "_segFromPrevStage.npz"))['data'][None]

                print(k, data.shape)
                data[-1][data[-1] == -1] = 0

                data_for_net = np.concatenate((data[:-1], to_one_hot(seg_from_prev_stage[0], range(1, self.num_classes))))

                softmax_pred = self.predict_preprocessed_data_return_seg_and_softmax(data_for_net,
                                                                                     do_mirroring=do_mirroring,
                                                                                     mirror_axes=mirror_axes,
                                                                                     use_sliding_window=use_sliding_window,
                                                                                     step_size=step_size,
                                                                                     use_gaussian=use_gaussian,
                                                                                     all_in_gpu=all_in_gpu,
                                                                                     mixed_precision=self.fp16)[1]

                softmax_pred = softmax_pred.transpose([0] + [i + 1 for i in self.transpose_backward])

                if save_softmax:
                    softmax_fname = join(output_folder, fname + ".npz")
                else:
                    softmax_fname = None

                if np.prod(softmax_pred.shape) > (2e9 / 4 * 0.85):  # *0.85 just to be save
                    np.save(join(output_folder, fname + ".npy"), softmax_pred)
                    softmax_pred = join(output_folder, fname + ".npy")

                results.append(export_pool.starmap_async(save_segmentation_nifti_from_softmax,
                                                         ((softmax_pred, join(output_folder, fname + ".nii.gz"),
                                                           properties, interpolation_order, None, None, None,
                                                           softmax_fname, None, force_separate_z,
                                                           interpolation_order_z),
                                                          )
                                                         )
                               )

            pred_gt_tuples.append([join(output_folder, fname + ".nii.gz"),
                                   join(self.gt_niftis_folder, fname + ".nii.gz")])

        _ = [i.get() for i in results]
        self.print_to_log_file("finished prediction")

        # evaluate raw predictions
        self.print_to_log_file("evaluation of raw predictions")
        task = self.dataset_directory.split("/")[-1]
        job_name = self.experiment_name
        _ = aggregate_scores(pred_gt_tuples, labels=list(range(self.num_classes)),
                             json_output_file=join(output_folder, "summary.json"),
                             json_name=job_name + " val tiled %s" % (str(use_sliding_window)),
                             json_task=task, num_threads=default_num_threads)

        self.print_to_log_file("determining postprocessing")
        determine_postprocessing(self.output_folder, self.gt_niftis_folder, validation_folder_name,
                                 final_subf_name=validation_folder_name + "_postprocessed", debug=debug)

        gt_nifti_folder = join(self.output_folder_base, "gt_niftis")
        maybe_mkdir_p(gt_nifti_folder)
        for f in subfiles(self.gt_niftis_folder, suffix=".nii.gz"):
            success = False
            attempts = 0
            e = None
            while not success and attempts < 10:
                try:
                    shutil.copy(f, gt_nifti_folder)
                    success = True
                except OSError as e:
                    attempts += 1
                    sleep(1)
            if not success:
                print("Could not copy gt nifti file %s into folder %s" % (f, gt_nifti_folder))
                if e is not None:
                    raise e

        # restore network deep supervision mode
        self.network.train(current_mode)
        self.network.do_ds = ds
