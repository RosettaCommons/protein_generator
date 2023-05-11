python ./inference.py \
    --num_designs 10 \
    --out examples/out/design \
    --pdb examples/pdbs/rsv5_5tpn.pdb \
    --contigs 0-25,A163-181,25-30 --T 25 --save_best_plddt
