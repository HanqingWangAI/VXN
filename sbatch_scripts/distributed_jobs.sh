#!/bin/bash

#SBATCH -n 16
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=8000
#SBATCH --gpus=rtx_3090:4
#SBATCH -o debug_logs/%j.out
#SBATCH --error=debug_logs/%j.err
# ip=$(hostname -I)
# echo $ip
# ifconfig 

# get IP address of the master node
ip=$(ifconfig | grep inet | grep -v 127.0.0.1 | grep -v 255.255.255.255 | grep -v inet6 | awk '{print $2}')
access_ip=$(echo $ip | awk '{print $1}')
fabric_ip=$(echo $ip | awk '{print $2}')

echo $ip

master_addr=$access_ip
nnodes=4
configs=("tf-hpn-_c_vln.yaml" "tf-hpn-_c_van.yaml" "tf-hpn-_c_ign.yaml" "tf-hpn-_c_ogn.yaml")
# configs=("tf-hpn-_c_van.yaml"  "tf-hpn-_c_vln.yaml" "tf-hpn-_c_ign.yaml" "tf-hpn-_c_ogn.yaml")

for i in 1 2 3
do
    # idx=($i)%4
    idx=1
    node_rank=$i master_addr=$master_addr nnodes=$nnodes config=${configs[idx]} sbatch $job_name 
done

# the master node
node_rank=0 master_addr=$master_addr nnodes=$nnodes config=${configs[1]} bash $job_name
