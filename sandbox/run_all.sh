#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=28
#SBATCH --partition=loki4
#SBATCH --job-name=k_srme_test
#SBATCH --time=04-00:00              # Runtime limit: Day-HH:MM

source $LTC_VENV
HEAD=/data2_1/jinvk/25_LTC/omat24
python cond_pll.py $head $SLURM_NTASKS 
