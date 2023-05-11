python ./inference.py \
    --num_designs 10 \
    --out examples/out/design \
    --contigs 100 \
    --T 25 --save_best_plddt \
    --helix_bias 0.01 --strand_bias 0.01 --loop_bias 0.0 
