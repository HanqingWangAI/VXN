#!/bin/bash
#SBATCH -n 16
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=8000
#SBATCH --gpus=rtx_3090:4
#SBATCH -o debug_logs/%j.out
#SBATCH --error=debug_logs/%j.err
# cd ..


export GLOG_minloglevel=2
export MAGNUM_LOG=quiet


echo "noderank ${node_rank} master addr ${master_addr} world size ${nnodes} config ${config}"

NCCL_IB_DISABLE=1 python -u -m torch.distributed.launch \
    --use_env \
    --nproc_per_node 4 \
    --nnodes ${nnodes} \
    --node_rank $node_rank \
    --master_addr $master_addr \
    --master_port 19988 \
    run.py \
    --exp-config vienna/config/r2r_waypoint/${config} \
    --run-type train