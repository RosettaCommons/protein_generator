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
    --pdb pdbs/rsv5_5tpn.pdb \
    --contigs 0-25,A163-181,25-30 --T 25 --save_best_plddt
