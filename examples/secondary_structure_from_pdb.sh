python ./inference.py \
    --num_designs 10 \
    --out examples/out/design \
    --contigs 110 \
    --T 25 --save_best_plddt \
    --dssp_pdb examples/pdbs/cd86.pdb

# FOR SECONDARY STRUCTURE:
#   X - mask
#   H - helix
#   E - strand
#   L - loop
