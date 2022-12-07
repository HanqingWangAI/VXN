# evaluate multi-tasks

configs=("tf-hpn-_c_vln_eval.yaml" "tf-hpn-_c_van_eval.yaml" "tf-hpn-_c_ign_eval.yaml" "tf-hpn-_c_ogn_eval.yaml")


for i in 0 1 2 3
do
    echo ${configs[i]}
    config=${configs[i]} sbatch sbatch_scripts/eval_job.sh
done