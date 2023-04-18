#!/bin/sh
#SBATCH --job-name=luka_cpu
#SBATCH --output=luka_cpu.out
#SBATCH --time=01:00:00       # job time limit
#SBATCH --nodes=1             # number of nodes
#SBATCH --ntasks-per-node=1   # number of tasks
#SBATCH --partition=cpu       # partition to run on nodes with CPUs only
#SBATCH --cpus-per-task=1     # number of allocated cores
#SBATCH --mem=16G             # memory allocation

#module load PyTorch/1.12.0-foss-2022a-CUDA-11.7.0

source ~/miniconda3/etc/profile.d/conda.sh # intialize conda
conda activate ai                 # activate the previously created environment

OUT_PATH=/d/hpc/projects/FRI/ldragar/

# Run the training script twice with different hyperparameters.
#srun --nodes=1 --exclusive --gpus=1 --ntasks=1 python train.py --gpu 0 --lr 1e-4 --epochs 50 --batch_size 64 --out_path $OUT_PATH &
#srun --nodes=1 --exclusive --gpus=1 --ntasks=1 python train.py --gpu 0 --lr 1e-3 --epochs 50 --batch_size 64 --out_path $OUT_PATH &

#srun --nodes=1 --exclusive --gpus=1 --ntasks=1 python 0_extract_faces.py --gpu 0 
wait
