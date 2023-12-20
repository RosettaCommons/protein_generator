python ./inference.py \
    --num_designs 10 \
    --out examples/out/design \
    --contigs 100 \
    --T 25 --save_best_plddt \
    --potentials aa_bias \
    --aa_composition W0.2 --potential_scale 1.75
