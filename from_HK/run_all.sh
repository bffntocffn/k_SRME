#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=28
#SBATCH --partition=loki4
##SBATCH --gres=gpu:1
##SBATCH --nodelist=n007
#SBATCH --job-name=k_srme_test
#SBATCH --time=04-00:00              # Runtime limit: Day-HH:MM

# export CUDA_DEVICE_ORDER=PCI_BUS_ID
# export CUDA_VISIBLE_DEVICES=1

model_name="7net_m3g_n"
model_path="/data2/shared_data/pretrained_experimental/${model_name}/checkpoint_best.pth"
python 1_test_srme_parallel.py $model_name $model_path $SLURM_NTASKS 
python 2_evaluate.py $model_name

