#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 1-00:00:00
#SBATCH --gpus=h100-80:1
#SBATCH --job-name=idl_diffusion_train
#SBATCH -A cis250233p
#SBATCH --output=slurm/train/%A_%a.out
#SBATCH --error=slurm/train/%A_%a.err
#SBATCH --export=ALL

# load conda
module load anaconda3/2024.10-1

# activate environment
conda activate idl_diffusion_env

nvidia-smi

export WANDB_API_KEY="2f02bf2d4bd05dc6abe9560736612ee323fa9b42"

# model training
cmd=`sed -n "${SLURM_ARRAY_TASK_ID}p" diffusion_run_train_cmds.txt`
eval $cmd
exit

