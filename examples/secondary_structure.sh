python ./inference.py \
    --num_designs 10 \
    --out examples/out/design \
    --contigs 100 \
    --T 25 --save_best_plddt \
    --secondary_structure XXXXXHHHHXXXLLLXXXXXXXXXXHHHHXXXLLLXXXXXXXXXXHHHHXXXLLLXXXXXXXXXXHHHHXXXLLLXXXXXXXXXXHHHHXXXLLLXXXXX

# FOR SECONDARY STRUCTURE:
#   X - mask
#   H - helix
#   E - strand
#   L - loop
