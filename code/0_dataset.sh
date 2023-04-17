#!/bin/sh
#SBATCH --job-name=extract_faces
#SBATCH --output=extract_faces_%j.out
#SBATCH --error=extract_faces_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --partition=gpu


#this code is used to extract the dataset of cropped faces from the original dataset
#and save them in the folder "dataset"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_env


input_root_path=/d/hpc/projects/FRI/ldragar/original_dataset/
output_root_path=./dataset/

for i in {1..3}
do
    echo extracting faces from C$i
    in="${input_root_path}C${i}"
    out="${output_root_path}C${i}"
    # echo $in
    # echo $out
    srun --nodes=1 --exclusive --gpus=1 --ntasks=1 -p gpu python 0_extract_faces.py --input_root_path $in --output_root_path $out --gpu_id 0 --scale 1.3 --id_num 61
done

echo "done"