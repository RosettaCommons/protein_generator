#!/bin/bash
#SBATCH -J seq_diff
#SBATCH -p gpu
#SBATCH --mem=8g
#SBATCH --gres=gpu:a6000:1
#SBATCH -o ./out/slurm/slurm_%j.out

source activate /software/conda/envs/SE3nv

srun python ../inference.py \
    --num_designs 10 \
    --out out/binder_design \
    --pdb pdbs/cd86.pdb \
    --T 25 --save_best_plddt \
    --contigs B1-110,0 25-75 \
    --hotspots B40,B32,B87,B96,B30
