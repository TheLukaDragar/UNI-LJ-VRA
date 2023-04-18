#!/bin/bash
#SBATCH --job-name=seqtrain
#SBATCH --nodes=2           # This needs to match Trainer(num_nodes=...)
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --ntasks-per-node=2  # This needs to match Trainer(devices=...)
#SBATCH --mem=0
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%j.out


# Load any necessary modules (e.g., CUDA, Python, etc.)


# Activate your Python environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_env
export WANDB__SERVICE_WAIT=300
# Run your training script

#check if the gpu is working
srun python train_convnext_SequecingWithzaporednimifrmi2nodes.py 


#salloc --nodes=1 --gres=gpu:2 --ntasks-per-node=2 --mem=0 --time=0-10:00:00 --cpus-per-task=8 --job-name=Interactive_GPU2 --partition=gpu 
