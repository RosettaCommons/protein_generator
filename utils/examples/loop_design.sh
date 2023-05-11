#!/bin/bash
#SBATCH -J seq_diff
#SBATCH -p gpu
#SBATCH --mem=8g
#SBATCH --gres=gpu:a6000:1
#SBATCH -o ./out/slurm/slurm_%j.out

source activate /software/conda/envs/SE3nv

srun python ../inference.py \
    --num_designs 10 \
    --pdb pdbs/G12D_manual_mut.pdb \
    --out out/ab_loop \
    --contigs A2-176,0 C7-16,0 H2-95,12-15,H111-116,0 L1-45,10-12,L56-107 \
    --T 25 --save_best_plddt --loop_design
