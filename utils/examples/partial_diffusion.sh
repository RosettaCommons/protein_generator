#!/bin/bash
#SBATCH -J seq_diff
#SBATCH -p gpu
#SBATCH --mem=8g
#SBATCH --gres=gpu:a6000:1
#SBATCH -o ./out/slurm/slurm_%j.out

source activate /software/conda/envs/SE3nv

srun python ../inference.py \
    --num_designs 10 \
    --pdb out/design_000.pdb \
    --trb out/design_000.trb \
    --out out/partial_diffusion_design \
    --contigs 0 --sampling_temp 0.3 --T 50 --save_best_plddt
