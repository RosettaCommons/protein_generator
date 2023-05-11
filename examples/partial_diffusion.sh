python ./inference.py \
    --num_designs 10 \
    --pdb examples/out/design_000000.pdb \
    --out examples/out/partial_diffusion_design \
    --contigs 38 --sampling_temp 0.3 --T 50 --save_best_plddt
