"""
RUN INFERENCE
"""
import sys, os, subprocess, pickle, time, json, argparse
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path = sys.path + [script_dir+'/utils/'] + [script_dir+'/model/']
from sampler import *

sampler_map = {'default':SEQDIFF_sampler, 'cleavage_foldswitch':cleavage_foldswitch_SAMPLER}


def get_args():
    """
    Parse command line args
    """
    parser = argparse.ArgumentParser()

    # design-related args
    parser.add_argument('--pdb','-p',dest='pdb', default=None,
            help='input protein')
    parser.add_argument('--sequence',type=str, default=None,
            help='input sequence to diffuse')
    parser.add_argument('--trb', default=None, help='input trb file for partial diffusion')
    parser.add_argument('--contigs', default='0', nargs='+',
            help='Pieces of input protein to keep ')
    parser.add_argument('--length',default=None,type=str,
            help='Specify length, or length range, you want the outputs. e.g. 100 or 95-105')
    parser.add_argument('--checkpoint', default=None,
            help='Checkpoint to pretrained RFold module')
    parser.add_argument('--inpaint_str', type=str, default=None, nargs='+',
         help='Predict the structure at these residues. Similar mask (and window), but is '\
              'specifically for structure.')
    parser.add_argument('--inpaint_seq', type=str, default=None, nargs='+',
         help='Predict the sequence at these residues. Similar mask (and window), but is '\
              'specifically for sequence.')
    parser.add_argument('--n_cycle', type=int, default=4,
            help='Number of recycles through RFold at each step')
    parser.add_argument('--tmpl_conf', type=str, default='1',
            help='1D confidence value for template residues')
    parser.add_argument('--num_designs', type=int, default=50,
            help='Number of designs to make')
    parser.add_argument('--start_num', type=int, default=0,
            help='Number of first design to output')
    parser.add_argument('--sampler',type=str, default='default',
            help='Type of sampler to use')

    # i/o args
    parser.add_argument('--out', default='./seqdiff',
            help='output directory and for files')
    parser.add_argument('--dump_pdb', default=True, action='store_true',
            help='Whether to dump pdb output')
    parser.add_argument('--dump_trb', default=True, action='store_true',
            help='Whether to dump trb files in output dir')
    parser.add_argument('--dump_npz', default=False, action='store_true',
            help='Whether to dump npz (disto/anglograms) files in output dir')
    parser.add_argument('--dump_all', default=False, action='store_true',
            help='If true, will dump all possible outputs to outdir')
    parser.add_argument('--input_json', type=str, default=None,
        help='JSON-formatted list of dictionaries, each containing command-line arguments for 1 '\
             'design.')
    parser.add_argument('--cautious', default=False, action='store_true',
            help='If true, will not run a design if output file already exists.')


    #diffusion args
    parser.add_argument('--T', default=25, type=int,
        help='Number of timesteps to use')
    parser.add_argument('--F', default=1, type=int,
        help='noise factor')
    parser.add_argument('--save_all_steps', default=False, action='store_true',
        help='Save individual steps during diffusion')
    parser.add_argument('--save_best_plddt', default=True, action='store_true',
        help='Save highest plddt structure only')
    parser.add_argument('--save_seqs', default=False, action='store_true',
        help='Save in and out seqs')
    parser.add_argument('--argmax_seq', default=False, action='store_true',
        help='Argmax seq after coming out of model')
    parser.add_argument('--noise_schedule', default='sqrt',
        help='Schedule type to add noise, default=cosine, could be [sqrt]')
    parser.add_argument('--sampling_temp', default=1.0, type=float,
        help='Temperature to sample input sequence to as a fraction of T, for partial diffusion')
    parser.add_argument('--loop_design', default=False, action='store_true',
        help='If this arg is passed the loop design checkpoint will be used')
    parser.add_argument('--symmetry', type=int, default=1,
        help='Integer specifying sequence repeat symmetry, e.g. 4 -> sequence composed of 4 identical repeats')
    parser.add_argument('--symmetry_cap', default=0, type=int,
        help='length for symmetry cap; assumes cap will be helix')
    parser.add_argument('--predict_symmetric', default=False, action='store_true',
        help='Predict explicit symmetrization after the last step')

    parser.add_argument('--frac_seq_to_weight', default=0.0, type=float,
        help='fraction of sequence to add AA weight bias too (will be randomly sampled)')
    parser.add_argument('--add_weight_every_n', default=1, type=int,
        help='frequency to add aa weight')
    parser.add_argument('--aa_weights_json', default=None, type=str,
        help='file path the JSON file of amino acid weighting to use during inference')
    parser.add_argument('--one_weight_per_position', default=False, action='store_true',
        help='only add weight to one aa type at each residue position (will randomly sample)')
    parser.add_argument('--aa_weight', default=None, type=str,
        help='weight string to use with --aa_spec for how to bias sequence')
    parser.add_argument('--aa_spec', default=None, type=str,
        help='how to bias sequence example XXXAXL where X is mask token')
    parser.add_argument('--aa_composition', default=None, type=str,
        help='aa composition specified by one letter aa code and fraction to represent in sequence ex. H0.2,K0.5')
    parser.add_argument('--d_t1d', default=24, type=int,
        help='t1d dimension that is compatible with specified checkpoint')
    parser.add_argument('--hotspots', default=None, type=str,
        help='specify hotspots to find i.e. B35,B44,B56')
    parser.add_argument('--secondary_structure', default=None, type=str,
        help='specified secondary structure string, H-helix, E-strand, L-loop, X-mask, i.e. XXXXXXHHHHHHXXXXLLLLXXXXXEEEEXXXXX')
    parser.add_argument('--helix_bias', default=0.0, type=float,
        help='percent of sequence to randomly bias toward helix')
    parser.add_argument('--strand_bias', default=0.0, type=float,
        help='percent of sequence to randomly bias toward strand')
    parser.add_argument('--loop_bias', default=0.0, type=float,
        help='percent of sequence to randomly bias toward loop')
    parser.add_argument('--dssp_pdb', default=None, type=str,
        help='input protein dssp')
    parser.add_argument('--scheduled_str_cond', default=False, action='store_true',
        help='if turned on will self condition on x fraction of the strcutre according to schedule (jake style)')
    parser.add_argument('--struc_cond', default=False, action='store_true',
        help='if turned on will struc condition on structure in sidneys style')
    parser.add_argument('--struc_cond_sc', default=False, action='store_true',
        help='if turned on will self condition on structure in sidneys style')
    parser.add_argument('--softmax_seqout', default=False, action='store_true',
        help='if turned on will softmax the Xo pred sequence before sampling next t')
    parser.add_argument('--clamp_seqout', default=False, action='store_true',
        help='if turned on will clamp the Xo pred sequence before sampling next t')
    parser.add_argument('--no_clamp_seqout_after', default=False, action='store_true',
        help='if turned on will clamp the Xo pred sequence before sampling next t')
    parser.add_argument('--save_args', default=False, action='store_true',
        help='will save the arguments used in a json file')

    # potential args
    parser.add_argument('--potential_scale', default=None, type=str,
        help='scale at which to guid the sequence potential')
    parser.add_argument('--potentials', default='', 
        help='list of potentials to use, must be paired with potenatial_scale e.g. aa_bias,solubility,charge')
    parser.add_argument('--hydrophobic_score', default='0', type=float, 
        help='Set GRAVY score to guide sequence towards. Default == 0.0')
    parser.add_argument('--hydrophobic_loss_type', default='complex', type=str,
        help='type of loss to compute when using hydrophobicity potential')
    parser.add_argument('--target_charge', default=0.0, type=float, 
        help='Set charge to guide sequence towards. Default == 0.0')
    parser.add_argument('--target_pH', default=7.4, type=float, 
        help='Set pH to calculate charge at. Default == 7.4')
    parser.add_argument('--charge_loss_type', default='complex', type=str,
        help='type of loss to use when using charge potential')
    parser.add_argument('--PSSM', default=None, type=str,
        help='PSSM as csv')

    # noise args
    parser.add_argument('--sample_distribution', default="normal", type=str,
        help='sample distribution for q_sample()')
    parser.add_argument('--sample_distribution_gmm_means', default=[-1.0, 1.0], nargs='+',
        help='sample distribution means for q_sample()')
    parser.add_argument('--sample_distribution_gmm_variances', default=[1.0, 1.0], nargs='+',
        help='sample distribution variances for q_sample()')



    return parser.parse_args()



def main():
    
    print(
    '''
            ██████╗ ██████╗  ██████╗ ████████╗███████╗██╗███╗   ██╗
            ██╔══██╗██╔══██╗██╔═══██╗╚══██╔══╝██╔════╝██║████╗  ██║
            ██████╔╝██████╔╝██║   ██║   ██║   █████╗  ██║██╔██╗ ██║
            ██╔═══╝ ██╔══██╗██║   ██║   ██║   ██╔══╝  ██║██║╚██╗██║
            ██║     ██║  ██║╚██████╔╝   ██║   ███████╗██║██║ ╚████║
            ╚═╝     ╚═╝  ╚═╝ ╚═════╝    ╚═╝   ╚══════╝╚═╝╚═╝  ╚═══╝
     ██████╗ ███████╗███╗   ██╗███████╗██████╗  █████╗ ████████╗ ██████╗ ██████╗ 
    ██╔════╝ ██╔════╝████╗  ██║██╔════╝██╔══██╗██╔══██╗╚══██╔══╝██╔═══██╗██╔══██╗
    ██║  ███╗█████╗  ██╔██╗ ██║█████╗  ██████╔╝███████║   ██║   ██║   ██║██████╔╝
    ██║   ██║██╔══╝  ██║╚██╗██║██╔══╝  ██╔══██╗██╔══██║   ██║   ██║   ██║██╔══██╗
    ╚██████╔╝███████╗██║ ╚████║███████╗██║  ██║██║  ██║   ██║   ╚██████╔╝██║  ██║
     ╚═════╝ ╚══════╝╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝
        

    '''
    )
    
    # parse args 
    args = get_args()
    
    # chose sampler
    chosen_sampler = sampler_map[args.sampler]
    print(f'using sampler: {args.sampler}')

    # initiate sampler class
    S = chosen_sampler(vars(args))    
    
    # get JSON args 
    if args.input_json is not None:
        with open(args.input_json) as f_json:
            argdicts = json.load(f_json)
        print(f'JSON args loaded {args.input_json}')
        # wrap argdicts in a list if not inputed as one
        if isinstance(argdicts,dict):
            argdicts = [argdicts]
        S.set_args(argdicts[0])
    else:
        # no json input, spoof list of argument dicts
        argdicts = [{}]

    # build model
    S.model_init()
    
    # diffuser init
    S.diffuser_init()


    for i_argdict, argdict in enumerate(argdicts):

        if args.input_json is not None:
            print(f'\nAdding argument dict {i_argdict} from input JSON ({len(argdicts)} total):')
            
            ### HERE IS WHERE ARGUMENTS SHOULD GET SET 
            S.set_args(argdict)
            S.diffuser_init()

        for i_des in range(S.args['start_num'], S.args['start_num']+S.args['num_designs']):

            out_prefix = f'{args.out}_{i_des:06}'
            
            if args.cautious and os.path.exists(out_prefix + '.pdb'):
                print(f'CAUTIOUS MODE: Skipping design because output file '\
                      f'{out_prefix + ".pdb"} already exists.')
                continue
            
            S.generate_sample()

if __name__ == '__main__':
    main()
