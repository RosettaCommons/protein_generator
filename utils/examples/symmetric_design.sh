#!/bin/bash
#SBATCH -J seq_diff
#SBATCH -p gpu
#SBATCH --mem=8g
#SBATCH --gres=gpu:a6000:1
#SBATCH -o ./out/slurm/slurm_%j.out

source activate /software/conda/envs/SE3nv

srun python ../inference.py \
    --num_designs 10 \
    --out out/symmetric_design \
    --contigs 25,0 25,0 25,0 \
    --T 50 \
    --save_best_plddt \
    --symmetry 3
