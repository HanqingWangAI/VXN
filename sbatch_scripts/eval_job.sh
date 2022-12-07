#!/bin/bash

#SBATCH -n 16
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=8000
#SBATCH --gpus=rtx_3090:1
#SBATCH -o debug_logs/%j.out


export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

echo ${config}
python run.py \
    --exp-config vlnce_baselines/config/r2r_waypoint/${config} \
    --run-type eval