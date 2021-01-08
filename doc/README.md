<!--
 * @Author: Niki
 * @Date: 2020-12-30 18:45:52
 * @Description: 
-->
## 数据集转换
```bash
nnUNet_convert_decathlon_task -i  $nnUNet_raw_data_base/nnUNet_raw_data/Task08_HepaticVessel
```

## 预处理
```bash
nnUNet_plan_and_preprocess -t 8
```

## 预训练模型下载
```bash
nnUNet_download_pretrained_model Task008_HepaticVessel
```

## 模型训练
8代表你的任务ID，4代表五折交叉验证（0代表一折）
### 单机单卡训练
```bash
nnUNet_train 3d_fullres nnUNetTrainerV2 8 0
nnUNet_train 3d_fullres nnUNetTrainerV2 8 4   # 一般常用五折交叉验证
```

### 单机多卡训练
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 nnUNet_train_DP 3d_fullres nnUNetTrainerV2_DP 8 0 -gpus 4    # 总batchsize数 = batchsize * GPUs, 每张卡都跑相同的batchsize
```

```bash
# DataParallel: 将输入一个 batch 的数据均分成多份，分别送到对应的 GPU 进行计算，各个 GPU 得到的梯度累加
CUDA_VISIBLE_DEVICES=0,1 nnUNet_train_DP 3d_fullres nnUNetTrainerV2_DP 8 0 -gpus 2 --dbs  # 总batchsize数不变， 每张卡均分batchsize，即minibatch = batchsize / GPUs
```

```bash
# DistributedDataParallel: 与 DataParallel 的单进程控制多 GPU 不同，DistributedDataParallel自动将训练分配给n个进程，分别在 n 个 GPU 上运行
# 每个进程对应一个独立的训练过程，且只对梯度等少量数据进行信息交换

cd /data0/JY/project/algorithm/nnUNet/nnunet

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --master_port=4321 --nproc_per_node=4 run/run_training_DDP.py 3d_fullres nnUNetTrainerV2_DDP 8 4 #

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --master_port=4321 --nproc_per_node=4 run/run_training_DDP.py 3d_fullres nnUNetTrainerV2_DDP 8 4 --dbs #
```

### 多机多卡训练
```bash

```

## 模型推理

```bash
nnUNet_find_best_configuration -m 2d 3d_fullres 3d_lowres 3d_cascade_fullres -t 8 --strict  # 确定最优模型

# nnUNet_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -t 8 -m CONFIGURATION --save_npz (option)
nnUNet_predict -i $nnUNet_raw_data_base/nnUNet_raw_data/Task008_HepaticVessel/imagesTs/ -o OUTPUT_DIRECTORY -t 8 -m 3d_fullres

```

