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

eva="./predictions/37orwro0/7327/"
convnext="./predictions/y23waiez/32585/"
weight=0.75
output="./predictions/combined/"

python combine_predictions.py --output $output --convnext $convnext --eva $eva --weight $weight
