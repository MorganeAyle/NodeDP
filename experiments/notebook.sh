#!/bin/bash
#SBATCH --partition gpu_all
#SBATCH -N 1 # number of nodes
#SBATCH --gres=gpu:1 # number of GPUs to be allocated
#SBATCH -t 1-00:00 # time after which the process will be killed (D-HH:MM)
#SBATCH -o "/nfs/homedirs/%u/slurm-output/slurm-%j.out"
#SBATCH --mem=16G # the memory (MB) that is allocated to the job. If your job exceeds this it will be killed
#SBATCH --qos=interactive # this qos ensures a very high priority but only one job per user can run under this mode.
#SBATCH --cpus-per-task=1
#SBATCH --exclude=mdsi-gpu01,mdsi-gpu02,gpu18,gpu19,gpu15,gpu16,gpu17

cd ${SLURM_SUBMIT_DIR}
echo Starting job ${SLURM_JOBID}
echo SLURM assigned me these nodes:
squeue -j ${SLURM_JOBID} -O nodelist | tail -n +2

# Activate your conda environment if necessary
source /nfs/students/ayle/miniconda3/etc/profile.d/conda.sh
conda activate gcn-dp

cd ..

export XDG_RUNTIME_DIR="" # Fixes Jupyter bug with read/write permissions https://github.com/jupyter/notebook/issues/1318
jupyter notebook --no-browser --ip=$(hostname).daml.in.tum.de