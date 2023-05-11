python ./inference.py \
    --num_designs 10 \
    --out examples/out/binder_design \
    --pdb examples/pdbs/cd86.pdb \
    --T 25 --save_best_plddt \
    --contigs B1-110,0 25-75 \
    --hotspots B40,B32,B87,B96,B30
