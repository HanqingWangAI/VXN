#!/bin/bash
# cores=16
# # memory_and_gpu="rusage[mem=16000,ngpus_excl_p=4]"
# gpus_per_node=4
# memory=16000
# gpu_mem=10000GB
# # gpu_type="select[gpu_mtotal0>=23000]"
# # gpu_type="select[gpu_mtotal0>10000]"
# runtime="48:00"
# log_dir=debug_logs


cores=$cores gpus_per_node=$gpus_per_node  memory=$memory gpu_mem=$gpu_mem runtime=$runtime log_dir=$log_dir job_name="sbatch_scripts/job.sh" sbatch sbatch_scripts/distributed_jobs.sh