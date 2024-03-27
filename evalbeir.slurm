#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=1000GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=msmarco
#SBATCH --mail-type=begin        
#SBATCH --mail-type=end
#SBATCH --mail-user=saumyam@princeton.edu


port=$(shuf -i 15000-16000 -n 1)

dataset="msmarco"
mpath="LR.0.05/checkpoint"
step="step-325"

name=$SLURM_JOB_ID-dataset

module purge
module load anaconda3/2023.3
conda activate nlp

python eval_beir.py \
        --model_name_or_path $mpath\
        --step_path $step\
        --dataset $dataset\

