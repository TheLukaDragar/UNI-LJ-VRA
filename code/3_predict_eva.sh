#!/bin/sh
#SBATCH --job-name=predict_convnext
#SBATCH --output=predict_convnext_%j.out
#SBATCH --error=predict_convnext_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --cpus-per-task=12
#SBATCH --time=0-10:00:00
#SBATCH --partition=gpu

#example salloc --nodes=1 --gres=gpu:2 --ntasks-per-node=2 --mem=0 --time=0-10:00:00 --cpus-per-task=12 --job-name=Interactive_GPU2 --partition=gpu 


source ~/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_env

seed=7327 #closest result seed or -1 for no seed
out_predictions_dir='./predictions/'
cp_id='37orwro0' 


#script is made to run on 1 node with 1 gpu
srun --nodes=1 --exclusive --gpus=1 --ntasks-per-node=1 --time=0-3:00:00 -p gpu python predict_eva.py --seed $seed --out_predictions_dir $out_predictions_dir --cp_id $cp_id --x_predictions 10
