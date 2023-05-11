#!/bin/bash
#SBATCH -J seq_diff
#SBATCH -p gpu
#SBATCH --mem=8g
#SBATCH --gres=gpu:a6000:1
#SBATCH -o ./out/slurm/slurm_%j.out

source activate /software/conda/envs/SE3nv

srun python ../inference.py \
    --num_designs 10 \
    --out out/design \
    --contigs 110 \
    --T 25 --save_best_plddt \
    --dssp_pdb ./pdbs/cd86.pdb

# FOR SECONDARY STRUCTURE:
#   X - mask
#   H - helix
#   E - strand
#   L - loop
