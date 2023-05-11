---
title: PROTEIN GENERATOR
emoji: 🐨
thumbnail: http://files.ipd.uw.edu/pub/sequence_diffusion/figs/diffusion_landscape.png
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 3.24.1
app_file: app.py
pinned: false
---

![Screenshot](./utils/figs/fig1.jpg)

## TLDR but I know how to use inpainting
Submit jobs with all the same args as [inpainting](https://git.ipd.uw.edu/jwatson3/proteininpainting) plus some new ones
- <code>--T</code> specify number of timesteps to use, good choices are 5,25,50,100 (try lower T first)
- <code>--save_best_plddt</code> recommended arg to save best plddt str/seq in trajectory
- <code>--loop_design</code> use when generating loops for binder design (or Ab loop design), will load finetuned checkpoint
- <code>--symmetry</code> integer to divide the input up with for symmetric update (input of X will divide the sequence into X symmetric motifs) 
- <code>--trb</code> specify trb when partially diffusing to use same mask as the design, must match pdb input (contigs will not get used)
- <code>--sampling_temp</code> fraction of diffusion trajectory to use for partial diffusion (default is 1.0 for fully diffused) values around 0.3 seem to give good diversity

**environment to use**
```
source activate /software/conda/envs/SE3nv
```
**example command**
```
python inference.py --pdb examples/pdbs/rsv5_5tpn.pdb --out examples/out/design \
        --contigs 0-25,A163-181,25-30 --T 25 --save_best_plddt
```
For jobs with inputs <75 residues it is feesible to run on CPUs. It's helpful to redesign output backbones with [MPNN](https://git.ipd.uw.edu/justas/proteinmpnn) (not sure if useful yet when using <code>--loop_design</code>). Check back for more updates.

## Getting started

Check out the templates in the example folder to see how you can set up jobs for the various design strategies

- [ ] [Motif or active-site scaffolding](examples/motif_scaffolding.sh)
- [ ] [Partial diffusion (design diversification)](examples/partial_diffusion.sh)
- [ ] [Anitbody / Loop design](examples/loop_design.sh)
- [ ] [Symmetric design](examples/symmetric_design.sh)

## Weighted Sequence design
Biasing the sequence by weighting certain amino acid types is a nice way to control and guide the diffusion process, generate interesting folds, and repeat units. It is possible to combine this technique with motif scaffolding as well. here are a few different ways to set up sequence potentials:

The <code>--aa_spec</code> argument used in combination with the <code>--aa_weight</code> allows you to specify the complete amino acid weighting pattern for a sequence. The pattern specified in aa_spec will be repeated along the entire length of the design.
 - <code>--aa_spec</code> base repeat unit to weight sequence with, X is used as a mask token, for example <code>--aa_spec XXAXXLXX</code> will generate solenoid folds like the one below
 - <code>--aa_weight</code> weights to assign for non masked residues in <code>aa_spec</code>, for example <code>--aa_weight 2,2</code> will weight alanine to 2 and leucine to 2

**Make solenoids with a little bias!**
<p align='center'>
   <img src='./utils/figs/fig3.jpg' width='250' height='250'>
</p>

**example job set up for sequene weighting**
```    
python inference.py \
    --num_designs 10 \ 
    --out examples/out/seq_bias \
    --contigs 100 --symmetry 5 \
    --T 25 --save_best_plddt \
    --aa_spec XXAXXLXX --aa_weight 2,2
```

In addition to the contigs above users can also use a disctionary to specify sequence weighting with [aa_weights](examples/aa_weights.json) for more generic uses. These weights can be specified with the <code>--aa_weights_json</code> arg and used in combination with the <code>--add_weight_every_n</code> arg or <code>--frac_seq_to_weight</code> arg. Each of these args defines where weights in the aa_weights dictionary will be applied to the sequence (you cannot specify both simultaneously). To add the weight every 5 residues use <code>--add_weight_every_n 5</code>. To add weight to a randomly sampled 40% of the sequence use <code>--frac_seq_to_weight 0.4</code>. If you add weight to multiple amino acid types in aa_weights, use the <code>--one_weight_per_position</code> flag to specify that a randomly sampled amino acid from aa_weight with a positive value should be chosen where the sequence bias is added. This allows the user to specify multiple amino acid types you want to upweight while ensuring to only bias for one type at each position, this usually is more effective.


## Motif and active site scaffolding
An example for motif scaffolding submission is written below, if you are inputing an active site with single residue inputs this can be specified in the contigs like <code>10-20,A10-10,20-30,A50-50,5-15</code> to scaffold just the 10th and 50th residues of chain A. Setting the model at higher T usually results in higher success rates, but it can still be useful to try problems out with just a few steps (T = 5, 15, or 25), before increasing the number of steps further. It is recommended to use [MPNN](https://git.ipd.uw.edu/justas/proteinmpnn) on the output backbones before alphafolding for validation.

```
python inference.py \
    --num_designs 10 \
    --out examples/out/design \
    --pdb examples/pdbs/rsv5_5tpn.pdb \
    --contigs 0-25,A163-181,25-30 --T 25 --save_best_plddt
```

## Partial diffusion
To sample diverse and highquality desing fast, it can be useful to run many designs with T=5, and then after MPNN and alphafold filtering partially diffuse the successful designs to generate more diversity around designs that seem to be working. By using the <code>--trb</code> flag the script will enter partial diffusion mode. With the <code>--T</code> flag you can specify the total number of steps inthe trajectory and with the <code>--sampling_temp</code> flag you can determine how far into the trajectory the inputs will be diffused. Setting the sampling temp to 1.0 would be full diffused. In this mode the contigs will be ignored, and the mask used from the original design will be used. 

```
python inference.py \
    --num_designs 10 \
    --pdb examples/out/design_000.pdb \
    --trb examples/out/design_000.trb \
    --out examples/out/partial_diffusion_design \
    --contigs 0 --sampling_temp 0.3 --T 50 --save_best_plddt
```


## Symmetric design
In symmetric design mode, the <code>--symmetry</code> flag is used to specify the number of partitions to make from the input sequence length. Each partition will be updated symmetric according to the first in the sequence. This requires that your sequence length (L) is divisible by the symmetry input. Symmetric motif scaffolding should be possible with the right contigs, but has not been experimented with yet.

```
python inference.py \
    --num_designs 10 \
    --pdb examples/pdbs/rsv5_5tpn.pdb \
    --out examples/out/symmetric_design \
    --contigs 25,0 25,0 25,0 \
    --T 50 --save_best_plddt --symmetry 3
```


## Antibody and loop design
Using the <code>--loop_desing</code> flag will load a version of the model finetuned on antibody CDR loops. This is useful if you are looking to design new CDR loops or are strcutred loops for binder design. It is helpful to run the designs with a target input too.

```
python inference.py \
    --num_designs 10 \
    --pdb examples/pdbs/G12D_manual_mut.pdb \
    --out examples/out/ab_loop \
    --contigs A2-176,0 C7-16,0 H2-95,12-15,H111-116,0 L1-45,10-12,L56-107 \
    --T 25 --save_best_plddt --loop_design
```



## About the model
Sequence diffusion is trained on the same dataset and uses the same architecture as RoseTTAFold. To train the model, a ground truth sequence is transformed into an Lx20 continuous space and gaussian noise is added to diffuse the sequence to the sampled timestep. To condition on structure and sequence, the structre for a motif is given and then corresponding sequence is denoised in the input. The rest of the structure is blackhole initialized. For each example the model is trained to predict Xo and losses are applied on the structure and sequence respectively. During training big T is set to 1000 steps, and a square root schedule is used to add noise.

![Screenshot](./utils/figs/fig2.jpg)


## Looking ahead
We are interested in problems where diffusing in sequence space is useful, if you would like to chat more or join in our effort for sequence diffusion come talk to Sidney or Jake!


## Acknowledgements
A project by Sidney Lisanza and Jake Gershon. Thanks to Sam Tipps for implementing symmetric sequence diffusion. Thank you to Minkyung Baek and Frank Dimaio for developing RoseTTAFold, Joe Watson and David Juergens for the developing inpainting inference script which the inference code is built on top of.

