#####################################################################
############# PROTEIN SEQUENCE DIFFUSION SAMPLER ####################
#####################################################################

import sys, os, subprocess, pickle, time, json
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path = sys.path + [script_dir+'/../model/'] + [script_dir+'/']
import shutil
import glob
import torch
import numpy as np
import copy
import json
import matplotlib.pyplot as plt
from torch import nn
import math
import re
import pickle
import pandas as pd
import random
from copy import deepcopy
import time
from collections import namedtuple
import math
from torch.nn.parallel import DistributedDataParallel as DDP
from RoseTTAFoldModel import RoseTTAFoldModule
from util import *
from inpainting_util import *
from kinematics import get_init_xyz, xyz_to_t2d
import parsers_inference as parsers
import diff_utils
import pickle
import pdb
from utils.calc_dssp import annotate_sse
from potentials import POTENTIALS
from diffusion import GaussianDiffusion_SEQDIFF

MODEL_PARAM ={
        "n_extra_block"   : 4,
        "n_main_block"    : 32,
        "n_ref_block"     : 4,
        "d_msa"           : 256,
        "d_msa_full"      : 64,
        "d_pair"          : 128,
        "d_templ"         : 64,
        "n_head_msa"      : 8,
        "n_head_pair"     : 4,
        "n_head_templ"    : 4,
        "d_hidden"        : 32,
        "d_hidden_templ"  : 32,
        "p_drop"       : 0.0
        }

SE3_PARAMS = {
        "num_layers_full"    : 1,
        "num_layers_topk" : 1,
        "num_channels"  : 32,
        "num_degrees"   : 2,
        "l0_in_features_full": 8,
        "l0_in_features_topk" : 64,
        "l0_out_features_full": 8,
        "l0_out_features_topk" : 64,
        "l1_in_features": 3,
        "l1_out_features": 2,
        "num_edge_features_full": 32,
        "num_edge_features_topk": 64,
        "div": 4,
        "n_heads": 4
        }

SE3_param_full = {}
SE3_param_topk = {}

for param, value in SE3_PARAMS.items():
    if "full" in param:
        SE3_param_full[param[:-5]] = value
    elif "topk" in param:
        SE3_param_topk[param[:-5]] = value
    else: # common arguments
        SE3_param_full[param] = value
        SE3_param_topk[param] = value
        
MODEL_PARAM['SE3_param_full'] = SE3_param_full
MODEL_PARAM['SE3_param_topk'] = SE3_param_topk

DEFAULT_CKPT = './SEQDIFF_221219_equalTASKS_nostrSELFCOND_mod30.pt'
t1d_29_CKPT = './SEQDIFF_230205_dssp_hotspots_25mask_EQtasks_mod30.pt'

class SEQDIFF_sampler:
    
    '''
        MODULAR SAMPLER FOR SEQUENCE DIFFUSION
        
        - the goal for modularizing this code is to make it as 
          easy as possible to edit and mix functions around 
        
        - in the base implementation here this can handle the standard
          inference mode with default passes through the model, different 
          forms of partial diffusion, and linear symmetry
    
    '''
    
    def __init__(self, args=None):
        '''
            set args and DEVICE as well as other default params
        '''
        self.args = args
        self.DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.conversion = 'ARNDCQEGHILKMFPSTWYVX-'
        self.dssp_dict = {'X':3,'H':0,'E':1,'L':2}
        self.MODEL_PARAM = MODEL_PARAM
        self.SE3_PARAMS = SE3_PARAMS
        self.SE3_param_full = SE3_param_full
        self.SE3_param_topk = SE3_param_topk
        self.use_potentials = False
        self.reset_design_num()
    
    def set_args(self, args):
        '''
            set new arguments if iterating through dictionary of multiple arguments
            
            # NOTE : args pertaining to the model will not be considered as this is
                     used to sample more efficiently without having to reload model for 
                     different sets of args
        '''
        self.args = args
        self.diffuser_init()
        if self.args['potentials'] not in ['', None]:
            self.potential_init()
        
    def reset_design_num(self):
        '''
            reset design num to 0
        '''
        self.design_num = 0
    
    def diffuser_init(self):
        '''
            set up diffuser object of GaussianDiffusion_SEQDIFF
        '''
        self.diffuser = GaussianDiffusion_SEQDIFF(T=self.args['T'],
                schedule=self.args['noise_schedule'],
                sample_distribution=self.args['sample_distribution'],
                sample_distribution_gmm_means=self.args['sample_distribution_gmm_means'],
                sample_distribution_gmm_variances=self.args['sample_distribution_gmm_variances'],
                ) 
        self.betas = self.diffuser.betas
        self.alphas = 1-self.betas
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0)
        
    def make_hotspot_features(self):
        '''
            set up hotspot features
        '''
        # initialize hotspot features to all 0s
        self.features['hotspot_feat'] = torch.zeros(self.features['L'])
        
        # if hotspots exist in args then make hotspot features
        if self.args['hotspots'] != None:
            self.features['hotspots'] = [(x[0],int(x[1:])) for x in self.args['hotspots'].split(',')]
            for n,x in enumerate(self.features['mappings']['complex_con_ref_pdb_idx']):
                if x in self.features['hotspots']:
                    self.features['hotspot_feat'][self.features['mappings']['complex_con_hal_idx0'][n]] = 1.0
                    
    def make_dssp_features(self):
        '''
            set up dssp features
        '''
        
        assert not ((self.args['secondary_structure'] != None) and (self.args['dssp_pdb'] != None)), \
               f'You are attempting to provide both dssp_pdb and/or secondary_secondary structure, please choose one or the other'
        
        # initialize with all zeros
        self.features['dssp_feat'] = torch.zeros(self.features['L'],4)
        
        if self.args['secondary_structure'] != None:
            
            self.features['secondary_structure'] = [self.dssp_dict[x.upper()] for x in self.args['secondary_structure']]
            
            assert len(self.features['secondary_structure']*self.features['sym'])+self.features['cap']*2 == self.features['L'], \
                f'You have specified a secondary structure string that does not match your design length'
            
            self.features['dssp_feat'] = torch.nn.functional.one_hot(
                torch.tensor(self.features['cap_dssp']+self.features['secondary_structure']*self.features['sym']+self.features['cap_dssp']),
                num_classes=4)
        
        elif self.args['dssp_pdb'] != None:
            dssp_xyz = torch.from_numpy(parsers.parse_pdb(self.args['dssp_pdb'])['xyz'][:,:,:])
            dssp_pdb = annotate_sse(np.array(dssp_xyz[:,1,:].squeeze()), percentage_mask=0)
            #we assume binder is chain A
            self.features['dssp_feat'][:dssp_pdb.shape[0]] = dssp_pdb
        
        elif (self.args['helix_bias'] + self.args['strand_bias'] + self.args['loop_bias']) > 0.0:

            tmp_mask = torch.rand(self.features['L']) < self.args['helix_bias']
            self.features['dssp_feat'][tmp_mask,0] = 1.0

            tmp_mask = torch.rand(self.features['L']) < self.args['strand_bias']
            self.features['dssp_feat'][tmp_mask,1] = 1.0

            tmp_mask = torch.rand(self.features['L']) < self.args['loop_bias']
            self.features['dssp_feat'][tmp_mask,2] = 1.0
        
        #contigs get mask label
        self.features['dssp_feat'][self.features['mask_str'][0],3] = 1.0
        #anything not labeled gets mask label
        mask_index = torch.where(torch.sum(self.features['dssp_feat'], dim=1) == 0)[0]
        self.features['dssp_feat'][mask_index,3] = 1.0
    
    def model_init(self):
        '''
            get model set up and choose checkpoint
        '''
       

        if self.args['checkpoint'] == None:
            self.args['checkpoint'] = DEFAULT_CKPT

        self.MODEL_PARAM['d_t1d'] = self.args['d_t1d']
        
        # decide based on input args what checkpoint to load
        if self.args['hotspots'] != None or self.args['secondary_structure'] != None \
            or (self.args['helix_bias'] + self.args['strand_bias'] + self.args['loop_bias']) > 0 \
            or self.args['dssp_pdb'] != None and self.args['checkpoint'] == DEFAULT_CKPT:
            
            self.MODEL_PARAM['d_t1d'] = 29
            print('You are using features only compatible with a newer model, switching checkpoint...')
            self.args['checkpoint'] = t1d_29_CKPT
        
        elif self.args['loop_design'] and self.args['checkpoint'] == DEFAULT_CKPT:
            print('Switched to loop design checkpoint')
            self.args['checkpoint'] = LOOP_CHECKPOINT
            
        # check to make sure checkpoint chosen exists
        if not os.path.exists(self.args['checkpoint']):
            print('WARNING: couldn\'t find checkpoint')
        
        if not os.path.exists(self.args['checkpoint']):
            raise Exception(f'MODEL NOT FOUND!\nTo down load models please run the following in the main directory:\nwget http://files.ipd.uw.edu/pub/sequence_diffusion/checkpoints/SEQDIFF_230205_dssp_hotspots_25mask_EQtasks_mod30.pt\nwget http://files.ipd.uw.edu/pub/sequence_diffusion/checkpoints/SEQDIFF_221219_equalTASKS_nostrSELFCOND_mod30.pt')

        self.ckpt = torch.load(self.args['checkpoint'], map_location=self.DEVICE)

        # check to see if [loader_param, model_param, loss_param] is in checkpoint
        #   if so then you are using v2 of inference with t2d bug fixed
        self.v2_mode = False
        if 'model_param' in self.ckpt.keys():
            print('You are running a new v2 model switching into v2 inference mode')
            self.v2_mode = True

            for k in self.MODEL_PARAM.keys():
                if k in self.ckpt['model_param'].keys():
                    self.MODEL_PARAM[k] = self.ckpt['model_param'][k]
                else:
                    print(f'no match for {k} in loaded model params')

        # make model and load checkpoint
        print('Loading model checkpoint...')
        self.model = RoseTTAFoldModule(**self.MODEL_PARAM).to(self.DEVICE)

        model_state = self.ckpt['model_state_dict']
        self.model.load_state_dict(model_state, strict=False)
        self.model.eval()
        print('Successfully loaded model checkpoint')
    
    def feature_init(self):
        '''
            featurize pdb and contigs and choose type of diffusion
        '''
        # initialize features dictionary for all example features
        self.features = {}
        
        # set up params
        self.loader_params = {'MAXCYCLE':self.args['n_cycle'],'TEMPERATURE':self.args['temperature'], 'DISTANCE':self.args['min_decoding_distance']}

        # symmetry
        self.features['sym'] = self.args['symmetry']
        self.features['cap'] = self.args['symmetry_cap']
        self.features['cap_dssp'] = [self.dssp_dict[x.upper()] for x in 'H'*self.features['cap']]
        if self.features['sym'] > 1:
            print(f"Input sequence symmetry {self.features['sym']}")
       
        assert (self.args['contigs'] in [('0'),(0),['0'],[0]] ) ^ (self.args['sequence'] in ['',None]),\
                f'You are specifying contigs ({self.args["contigs"]}) and sequence ({self.args["sequence"]})  (or neither), please specify one or the other'
        
        # initialize trb dictionary
        self.features['trb_d'] = {}
        
        if self.args['pdb'] == None and self.args['sequence'] not in ['', None]:
            print('Preparing sequence input')

            allowable_aas = [x for x in self.conversion[:-1]]
            for x in self.args['sequence']: assert x in allowable_aas, f'Amino Acid {x} is undefinded, please only use standart 20 AAs'
            self.features['seq'] = torch.tensor([self.conversion.index(x) for x in self.args['sequence']])
            self.features['xyz_t'] = torch.full((1,1,len(self.args['sequence']),27,3), np.nan)

            self.features['mask_str'] = torch.zeros(len(self.args['sequence'])).long()[None,:].bool()
            self.features['mask_seq'] = torch.tensor([0 if x == 'X' else 1 for x in self.args['sequence']]).long()[None,:].bool()
            self.features['blank_mask'] = torch.ones(self.features['mask_str'].size()[-1])[None,:].bool()

            self.features['idx_pdb'] = torch.tensor([i for i in range(len(self.args['sequence']))])[None,:]
            conf_1d = torch.ones_like(self.features['seq'])
            conf_1d[~self.features['mask_str'][0]] = 0
            self.features['seq_hot'], self.features['msa'], \
                self.features['msa_hot'], self.features['msa_extra_hot'], _ = MSAFeaturize_fixbb(self.features['seq'][None,:],self.loader_params)
            self.features['t1d'] = TemplFeaturizeFixbb(self.features['seq'], conf_1d=conf_1d)[None,None,:]
            self.features['seq_hot'] = self.features['seq_hot'].unsqueeze(dim=0)
            self.features['msa'] = self.features['msa'].unsqueeze(dim=0)
            self.features['msa_hot'] = self.features['msa_hot'].unsqueeze(dim=0)
            self.features['msa_extra_hot'] = self.features['msa_extra_hot'].unsqueeze(dim=0)
        
            self.max_t = int(self.args['T']*self.args['sampling_temp'])
            
            self.features['pdb_idx'] = [('A',i+1) for i in range(len(self.args['sequence']))]
            self.features['trb_d']['inpaint_str'] = self.features['mask_str'][0]
            self.features['trb_d']['inpaint_seq'] = self.features['mask_seq'][0]

        else:
            
            assert not (self.args['pdb'] == None and self.args['sampling_temp'] != 1.0),\
                    f'You must specify a pdb if attempting to use contigs with partial diffusion, else partially diffuse sequence input'
            
            if self.args['pdb'] == None:
                self.features['parsed_pdb'] = {'seq':np.zeros((1),'int64'),
                                                'xyz':np.zeros((1,27,3),'float32'),
                                                'idx':np.zeros((1),'int64'),
                                                'mask':np.zeros((1,27), bool),
                                                'pdb_idx':['A',1]}
            else:
                # parse input pdb
                self.features['parsed_pdb'] = parsers.parse_pdb(self.args['pdb'])
            
            # generate contig map
            self.features['rm'] = ContigMap(self.features['parsed_pdb'], self.args['contigs'], 
                                            self.args['inpaint_seq'], self.args['inpaint_str'], 
                                            self.args['length'], self.args['ref_idx'],
                                            self.args['hal_idx'], self.args['idx_rf'], 
                                            self.args['inpaint_seq_tensor'], self.args['inpaint_str_tensor'])
            self.features['mappings'] = get_mappings(self.features['rm'])

            self.features['pdb_idx'] = self.features['rm'].hal

                ### PREPARE FEATURES DEPENDING ON TYPE OF ARGUMENTS SPECIFIED ###
            
            # FULL DIFFUSION MODE
            if self.args['trb'] == None and self.args['sampling_temp'] == 1.0:
                # process contigs and generate masks
                self.features['mask_str'] = torch.from_numpy(self.features['rm'].inpaint_str)[None,:]
                self.features['mask_seq'] = torch.from_numpy(self.features['rm'].inpaint_seq)[None,:]
                self.features['blank_mask'] = torch.ones(self.features['mask_str'].size()[-1])[None,:].bool()

                seq_input = torch.from_numpy(self.features['parsed_pdb']['seq'])
                xyz_input = torch.from_numpy(self.features['parsed_pdb']['xyz'][:,:,:])

                self.features['xyz_t'] = torch.full((1,1,len(self.features['rm'].ref),27,3), np.nan)
                self.features['xyz_t'][:,:,self.features['rm'].hal_idx0,:14,:] = xyz_input[self.features['rm'].ref_idx0,:14,:][None, None,...]
                self.features['seq'] = torch.full((1,len(self.features['rm'].ref)),20).squeeze()
                self.features['seq'][self.features['rm'].hal_idx0] = seq_input[self.features['rm'].ref_idx0]
                
                # template confidence 
                conf_1d = torch.ones_like(self.features['seq'])*float(self.args['tmpl_conf'])
                conf_1d[~self.features['mask_str'][0]] = 0 # zero confidence for places where structure is masked
                seq_masktok = torch.where(self.features['seq'] == 20, 21, self.features['seq'])

                # Get sequence and MSA input features 
                self.features['seq_hot'], self.features['msa'], \
                    self.features['msa_hot'], self.features['msa_extra_hot'], _ = MSAFeaturize_fixbb(seq_masktok[None,:],self.loader_params)
                self.features['t1d'] = TemplFeaturizeFixbb(self.features['seq'], conf_1d=conf_1d)[None,None,:]
                self.features['idx_pdb'] = torch.from_numpy(np.array(self.features['rm'].rf)).int()[None,:]
                self.features['seq_hot'] = self.features['seq_hot'].unsqueeze(dim=0)
                self.features['msa'] = self.features['msa'].unsqueeze(dim=0)
                self.features['msa_hot'] = self.features['msa_hot'].unsqueeze(dim=0)
                self.features['msa_extra_hot'] = self.features['msa_extra_hot'].unsqueeze(dim=0)

                self.max_t = int(self.args['T']*self.args['sampling_temp'])
            
            # PARTIAL DIFFUSION MODE, NO INPUT TRB
            elif self.args['trb'] != None:
                print('Running in partial diffusion mode . . .')
                self.features['trb_d'] = np.load(self.args['trb'], allow_pickle=True)
                self.features['mask_str'] = self.features['trb_d']['inpaint_str'].clone()[None,:]
                self.features['mask_seq'] = self.features['trb_d']['inpaint_seq'].clone()[None,:]
                self.features['blank_mask'] = torch.ones(self.features['mask_str'].size()[-1])[None,:].bool()

                self.features['seq'] = torch.from_numpy(self.features['parsed_pdb']['seq'])
                self.features['xyz_t'] = torch.from_numpy(self.features['parsed_pdb']['xyz'][:,:,:])[None,None,...]

                if self.features['mask_seq'].shape[1] == 0:
                    self.features['mask_seq'] = torch.zeros(self.features['seq'].shape[0])[None].bool()
                if self.features['mask_str'].shape[1] == 0:
                    self.features['mask_str'] = torch.zeros(self.features['xyz_t'].shape[2])[None].bool()

                idx_pdb = []
                chains_used = [self.features['parsed_pdb']['pdb_idx'][0][0]]
                idx_jump = 0
                for i,x in enumerate(self.features['parsed_pdb']['pdb_idx']):
                    if x[0] not in chains_used:
                        chains_used.append(x[0])
                        idx_jump += 200
                    idx_pdb.append(idx_jump+i)
                    
                self.features['idx_pdb'] = torch.tensor(idx_pdb)[None,:]
                conf_1d = torch.ones_like(self.features['seq'])
                conf_1d[~self.features['mask_str'][0]] = 0
                self.features['seq_hot'], self.features['msa'], \
                    self.features['msa_hot'], self.features['msa_extra_hot'], _ = MSAFeaturize_fixbb(self.features['seq'][None,:],self.loader_params)
                self.features['t1d'] = TemplFeaturizeFixbb(self.features['seq'], conf_1d=conf_1d)[None,None,:]
                self.features['seq_hot'] = self.features['seq_hot'].unsqueeze(dim=0)
                self.features['msa'] = self.features['msa'].unsqueeze(dim=0)
                self.features['msa_hot'] = self.features['msa_hot'].unsqueeze(dim=0)
                self.features['msa_extra_hot'] = self.features['msa_extra_hot'].unsqueeze(dim=0)
                
                self.max_t = int(self.args['T']*self.args['sampling_temp'])
                
            else:
                print('running in partial diffusion mode, with no trb input, diffusing whole input')
                self.features['seq'] = torch.from_numpy(self.features['parsed_pdb']['seq'])
                self.features['xyz_t'] = torch.from_numpy(self.features['parsed_pdb']['xyz'][:,:,:])[None,None,...]

                if self.args['contigs'] in [('0'),(0),['0'],[0]]:
                    print('no contigs given partially diffusing everything')
                    self.features['mask_str'] = torch.zeros(self.features['xyz_t'].shape[2]).long()[None,:].bool()
                    self.features['mask_seq'] = torch.zeros(self.features['seq'].shape[0]).long()[None,:].bool()
                    self.features['blank_mask'] = torch.ones(self.features['mask_str'].size()[-1])[None,:].bool()
                else:
                    print('found contigs setting up masking for partial diffusion')
                    self.features['mask_str'] = torch.from_numpy(self.features['rm'].inpaint_str)[None,:]
                    self.features['mask_seq'] = torch.from_numpy(self.features['rm'].inpaint_seq)[None,:]
                    self.features['blank_mask'] = torch.ones(self.features['mask_str'].size()[-1])[None,:].bool()

                idx_pdb = []
                chains_used = [self.features['parsed_pdb']['pdb_idx'][0][0]]
                idx_jump = 0
                for i,x in enumerate(self.features['parsed_pdb']['pdb_idx']):
                    if x[0] not in chains_used:
                        chains_used.append(x[0])
                        idx_jump += 200
                    idx_pdb.append(idx_jump+i)

                self.features['idx_pdb'] = torch.tensor(idx_pdb)[None,:]
                conf_1d = torch.ones_like(self.features['seq'])
                conf_1d[~self.features['mask_str'][0]] = 0
                self.features['seq_hot'], self.features['msa'], \
                    self.features['msa_hot'], self.features['msa_extra_hot'], _ = MSAFeaturize_fixbb(self.features['seq'][None,:],self.loader_params)
                self.features['t1d'] = TemplFeaturizeFixbb(self.features['seq'], conf_1d=conf_1d)[None,None,:]
                self.features['seq_hot'] = self.features['seq_hot'].unsqueeze(dim=0)
                self.features['msa'] = self.features['msa'].unsqueeze(dim=0)
                self.features['msa_hot'] = self.features['msa_hot'].unsqueeze(dim=0)
                self.features['msa_extra_hot'] = self.features['msa_extra_hot'].unsqueeze(dim=0)
            
                self.max_t = int(self.args['T']*self.args['sampling_temp'])
            
        # set L
        self.features['L'] = self.features['seq'].shape[0]
        
    def potential_init(self):
        '''
            initialize potential functions being used and return list of potentails
        '''
        
        potentials = self.args['potentials'].split(',')
        potential_scale = [float(x) for x in self.args['potential_scale'].split(',')]
        assert len(potentials) == len(potential_scale), \
            f'Please make sure number of potentials matches potential scales specified'
        
        self.potential_list = []
        for p,s in zip(potentials, potential_scale):
            assert p in POTENTIALS.keys(), \
                f'The potential specified: {p} , does not match into POTENTIALS dictionary in potentials.py'
            print(f'Using potential: {p}')
            self.potential_list.append(POTENTIALS[p](self.args, self.features, s, self.DEVICE))
        
        self.use_potentials = True
        
    def setup(self, init_model=True):
        '''
            run init model and init features to get everything prepped to go into model
        '''
        
        # initialize features
        self.feature_init()
        
        # initialize potential
        if self.args['potentials'] not in ['', None]:  
            self.potential_init()
        
        # make hostspot features
        self.make_hotspot_features()
        
        # make dssp features
        self.make_dssp_features()
        
        # diffuse sequence and mask features
        self.features['seq'], self.features['msa_masked'], \
        self.features['msa_full'], self.features['xyz_t'], self.features['t1d'], \
        self.features['seq_diffused'] = diff_utils.mask_inputs(self.features['seq_hot'],
                                                               self.features['msa_hot'],
                                                               self.features['msa_extra_hot'],
                                                               self.features['xyz_t'],
                                                               self.features['t1d'],
                                                               input_seq_mask=self.features['mask_seq'],
                                                               input_str_mask=self.features['mask_str'],
                                                               input_t1dconf_mask=self.features['blank_mask'],
                                                               diffuser=self.diffuser,
                                                               t=self.max_t,
                                                               MODEL_PARAM=self.MODEL_PARAM,
                                                               hotspots=self.features['hotspot_feat'],
                                                               dssp=self.features['dssp_feat'],
                                                               v2_mode=self.v2_mode)
        
        
        # move features to device 
        self.features['idx_pdb'] = self.features['idx_pdb'].long().to(self.DEVICE, non_blocking=True) # (B, L)
        self.features['mask_str'] = self.features['mask_str'][None].to(self.DEVICE, non_blocking=True) # (B, L)
        self.features['xyz_t'] = self.features['xyz_t'][None].to(self.DEVICE, non_blocking=True)
        self.features['t1d'] = self.features['t1d'][None].to(self.DEVICE, non_blocking=True)
        self.features['seq'] = self.features['seq'][None].type(torch.float32).to(self.DEVICE, non_blocking=True)
        self.features['msa'] = self.features['msa'].type(torch.float32).to(self.DEVICE, non_blocking=True)
        self.features['msa_masked'] = self.features['msa_masked'][None].type(torch.float32).to(self.DEVICE, non_blocking=True)
        self.features['msa_full'] = self.features['msa_full'][None].type(torch.float32).to(self.DEVICE, non_blocking=True)
        self.ti_dev =  torsion_indices.to(self.DEVICE, non_blocking=True)
        self.ti_flip = torsion_can_flip.to(self.DEVICE, non_blocking=True)
        self.ang_ref = reference_angles.to(self.DEVICE, non_blocking=True)
        self.features['xyz_prev'] = torch.clone(self.features['xyz_t'][0])
        self.features['seq_diffused'] = self.features['seq_diffused'][None].to(self.DEVICE, non_blocking=True)
        self.features['B'], _, self.features['N'], self.features['L'] = self.features['msa'].shape
        self.features['t2d'] = xyz_to_t2d(self.features['xyz_t'])

        # get alphas
        self.features['alpha'], self.features['alpha_t'] = diff_utils.get_alphas(self.features['t1d'], self.features['xyz_t'], 
                                                                                 self.features['B'], self.features['L'], 
                                                                                 self.ti_dev, self.ti_flip, self.ang_ref)

        # processing template coordinates
        self.features['xyz_t'] = get_init_xyz(self.features['xyz_t'])
        self.features['xyz_prev'] = get_init_xyz(self.features['xyz_prev'][:,None]).reshape(self.features['B'], self.features['L'], 27, 3)
        
        # initialize extra features to none
        self.features['xyz'] = None
        self.features['pred_lddt'] = None
        self.features['logit_s'] = None
        self.features['logit_aa_s'] = None
        self.features['best_plddt'] = 0
        self.features['best_pred_lddt'] = torch.zeros_like(self.features['mask_str'])[0].float()
        self.features['msa_prev'] = None
        self.features['pair_prev'] = None
        self.features['state_prev'] = None

        
    def symmetrize_seq(self, x):
        '''
            symmetrize x according sym in features
        '''
        assert (self.features['L']-self.features['cap']*2) % self.features['sym'] == 0, f'symmetry does not match for input length'
        assert x.shape[0] == self.features['L'], f'make sure that dimension 0 of input matches to L'
        
        if self.features['cap'] > 0:
            n_cap = torch.clone(x[:self.features['cap']])
            c_cap = torch.clone(x[-self.features['cap']+1:])
            sym_x = torch.clone(x[self.features['cap']:self.features['L']//self.features['sym']]).repeat(self.features['sym'],1)
            
            return torch.cat([n_cap,sym_x,c_cap], dim=0)
        else:
            return torch.clone(x[:self.features['L']//self.features['sym']]).repeat(self.features['sym'],1)

    def predict_x(self):
        '''
            take step using X_t-1 features to predict Xo
        '''
        self.features['seq'], \
        self.features['xyz'], \
        self.features['pred_lddt'], \
        self.features['logit_s'], \
        self.features['logit_aa_s'], \
        self.features['alpha'], \
        self.features['msa_prev'], \
        self.features['pair_prev'], \
        self.features['state_prev'] \
        = diff_utils.take_step_nostate(self.model,
        self.features['msa_masked'], 
        self.features['msa_full'], 
        self.features['seq'], 
        self.features['t1d'], 
        self.features['t2d'], 
        self.features['idx_pdb'], 
        self.args['n_cycle'],
        self.features['xyz_prev'], 
        self.features['alpha'], 
        self.features['xyz_t'],
        self.features['alpha_t'],
        self.features['seq_diffused'],
        self.features['msa_prev'], 
        self.features['pair_prev'],
        self.features['state_prev'])
    
    def self_condition_seq(self):
        '''
            get previous logits and set at t1d template
        '''
        self.features['t1d'][:,:,:,:21] = self.features['logit_aa_s'][0,:21,:].permute(1,0)

    def self_condition_str_scheduled(self):
        '''
            unmask random fraction of residues according to timestep
        '''
        print('self_conditioning on strcuture')
        xyz_prev_template = torch.clone(self.features['xyz'])[None]
        self_conditioning_mask = torch.rand(self.features['L']) < self.diffuser.alphas_cumprod[t]
        xyz_prev_template[:,:,~self_conditioning_mask] = float('nan')
        xyz_prev_template[:,:,self.features['mask_str'][0][0]] = float('nan')
        xyz_prev_template[:,:,:,3:] = float('nan')
        t2d_sc = xyz_to_t2d(xyz_prev_template)

        xyz_t_sc = torch.zeros_like(self.features['xyz_t'][:,:1])
        xyz_t_sc[:,:,:,:3] = xyz_prev_template[:,:,:,:3]
        xyz_t_sc[:,:,:,3:] = float('nan')

        t1d_sc = torch.clone(self.features['t1d'][:,:1])
        t1d_sc[:,:,~self_conditioning_mask] = 0
        t1d_sc[:,:,mask_str[0][0]] = 0

        self.features['t1d'] = torch.cat([self.features['t1d'][:,:1],t1d_sc], dim=1)
        self.features['t2d'] = torch.cat([self.features['t2d'][:,:1],t2d_sc], dim=1)
        self.features['xyz_t'] = torch.cat([self.features['xyz_t'][:,:1],xyz_t_sc], dim=1)

        self.features['alpha'], self.features['alpha_t'] = diff_utils.get_alphas(self.features['t1d'], self.features['xyz_t'], 
                                                                                 self.features['B'], self.features['L'], 
                                                                                 self.ti_dev, self.ti_flip, self.ang_ref)
        self.features['xyz_t'] = get_init_xyz(self.features['xyz_t'])

    
    def self_condition_str(self):
        '''
            conditioining on strucutre in NAR way
        '''
        print("conditioning on structure for NAR structure noising")
        xyz_t_str_sc           = torch.zeros_like(self.features['xyz_t'][:,:1])
        xyz_t_str_sc[:,:,:,:3] = torch.clone(self.features['xyz'])[None]
        xyz_t_str_sc[:,:,:,3:] = float('nan')
        t2d_str_sc             = xyz_to_t2d(self.features['xyz_t'])
        t1d_str_sc             = torch.clone(self.features['t1d'])

        self.features['xyz_t'] = torch.cat([self.features['xyz_t'],xyz_t_str_sc], dim=1)
        self.features['t2d']   = torch.cat([self.features['t2d'],t2d_str_sc], dim=1)
        self.features['t1d']   = torch.cat([self.features['t1d'],t1d_str_sc], dim=1)
    
    def save_step(self):
        '''
            add step to trajectory dictionary
        '''
        self.trajectory[f'step{self.t}'] = (self.features['xyz'].squeeze().detach().cpu(), 
                                            self.features['logit_aa_s'][0,:21,:].permute(1,0).detach().cpu(), 
                                            self.features['seq_diffused'][0,:,:21].detach().cpu())
    
    def noise_x(self):
        '''
            get X_t-1 from predicted Xo
        '''
        # sample x_t-1
        self.features['post_mean'] = self.diffuser.q_sample(self.features['seq_out'], self.t, DEVICE=self.DEVICE)

        if self.features['sym'] > 1:
            self.features['post_mean'] = self.symmetrize_seq(self.features['post_mean'])

        # update seq and masks
        self.features['seq_diffused'][0,~self.features['mask_seq'][0],:21] = self.features['post_mean'][~self.features['mask_seq'][0],...]
        self.features['seq_diffused'][0,:,21] = 0.0
        
        # did not know we were clamping seq
        self.features['seq_diffused'] = torch.clamp(self.features['seq_diffused'], min=-3, max=3)
        
        # match other features to seq diffused
        self.features['seq'] = torch.argmax(self.features['seq_diffused'], dim=-1)[None]
        self.features['msa_masked'][:,:,:,:,:22] = self.features['seq_diffused']
        self.features['msa_masked'][:,:,:,:,22:44] = self.features['seq_diffused']
        self.features['msa_full'][:,:,:,:,:22] = self.features['seq_diffused']
        self.features['t1d'][:1,:,:,22] = 1-int(self.t)/self.args['T']

        
    def apply_potentials(self):
        '''
            apply potentials
        '''
        
        grads = torch.zeros_like(self.features['seq_out'])
        for p in self.potential_list:
            grads += p.get_gradients(self.features['seq_out'])
        
        self.features['seq_out'] += (grads/len(self.potential_list))
        
    def generate_sample(self):
        '''
            sample from the model 
            
            this function runs the full sampling loop
        '''
        # setup example
        self.setup()
        
        # start time
        self.start_time = time.time()
        
        # set up dictionary to save at each step in trajectory
        self.trajectory = {}
        
        # set out prefix
        self.out_prefix = self.args['out']+f'_{self.design_num:06}'
        print(f'Generating sample {self.design_num:06} ...')
        
        # main sampling loop
        for j in range(self.max_t):
            self.t = torch.tensor(self.max_t-j-1).to(self.DEVICE)
            
            # run features through the model to get X_o prediction
            self.predict_x()
            
            # save step
            if self.args['save_all_steps']:
                self.save_step()
            
            # get seq out
            self.features['seq_out'] = torch.permute(self.features['logit_aa_s'][0], (1,0))
            
            # save best seq
            if self.features['pred_lddt'].mean().item() > self.features['best_plddt']:
                self.features['best_seq'] = torch.argmax(torch.clone(self.features['seq_out']), dim=-1)
                self.features['best_pred_lddt'] = torch.clone(self.features['pred_lddt'])
                self.features['best_xyz'] = torch.clone(self.features['xyz'])
                self.features['best_plddt'] = self.features['pred_lddt'].mean().item()
            
            # self condition on sequence
            self.self_condition_seq()
            
            # self condition on structure
            if self.args['scheduled_str_cond']:
                self.self_condition_str_scheduled()
            if self.args['struc_cond_sc']:
                self.self_condition_str()
            
            # sequence alterations
            if self.args['softmax_seqout']:
                self.features['seq_out'] = torch.softmax(self.features['seq_out'],dim=-1)*2-1
            if self.args['clamp_seqout']:
                self.features['seq_out'] = torch.clamp(self.features['seq_out'], 
                                                       min=-((1/self.diffuser.alphas_cumprod[t])*0.25+5), 
                                                       max=((1/self.diffuser.alphas_cumprod[t])*0.25+5))
            
            # apply potentials
            if self.use_potentials:
                self.apply_potentials()
            
            # noise to X_t-1
            if self.t != 0:
                self.noise_x()
            
            print(''.join([self.conversion[i] for i in torch.argmax(self.features['seq_out'],dim=-1)]))
            print ("    TIMESTEP [%02d/%02d]   |   current PLDDT: %.4f   <<  >>   best PLDDT: %.4f"%(
                    self.t+1, self.args['T'], self.features['pred_lddt'].mean().item(), 
                    self.features['best_pred_lddt'].mean().item()))
        
        # record time
        self.delta_time = time.time() - self.start_time
        
        # save outputs
        self.save_outputs()
        
        # increment design num
        self.design_num += 1
        
        print(f'Finished design {self.out_prefix} in {self.delta_time/60:.2f} minutes.')
        
    def save_outputs(self):
        '''
            save the outputs from the model
        '''
        # save trajectory
        if self.args['save_all_steps']:
            fname = f'{self.out_prefix}_trajectory.pt'
            torch.save(self.trajectory, fname)
        
        # get items from best plddt step
        if self.args['save_best_plddt']:
            self.features['seq'] = torch.clone(self.features['best_seq'])
            self.features['pred_lddt'] = torch.clone(self.features['best_pred_lddt'])
            self.features['xyz'] = torch.clone(self.features['best_xyz'])
        
        # get chain IDs
        if (self.args['sampling_temp'] == 1.0 and self.args['trb'] == None) or (self.args['sequence'] not in ['',None]):
            chain_ids = [i[0] for i in self.features['pdb_idx']]
        elif self.args['dump_pdb']:
            chain_ids = [i[0] for i in self.features['parsed_pdb']['pdb_idx']]
        
        # write output pdb
        fname = self.out_prefix + '.pdb'
        if len(self.features['seq'].shape) == 2:
            self.features['seq'] = self.features['seq'].squeeze()
        write_pdb(fname, 
                  self.features['seq'].type(torch.int64), 
                  self.features['xyz'].squeeze(), 
                  Bfacts=self.features['pred_lddt'].squeeze(), 
                  chains=chain_ids)
        
        if self.args['dump_trb']:
            self.save_trb()
        
        if self.args['save_args']:
            self.save_args()

    def save_trb(self):
        '''
            save trb file
        '''
        
        lddt = self.features['pred_lddt'].squeeze().cpu().numpy()
        strmasktemp = self.features['mask_str'].squeeze().cpu().numpy()

        partial_lddt = [lddt[i] for i in range(np.shape(strmasktemp)[0]) if strmasktemp[i] == 0]
        trb = {}
        trb['lddt'] = lddt
        trb['inpaint_lddt'] = partial_lddt
        trb['contigs'] = self.args['contigs']
        trb['device'] = self.DEVICE
        trb['time'] = self.delta_time
        trb['args'] = self.args
        
        if self.args['sequence'] != None:
            for key, value in self.features['trb_d'].items():
                trb[key] = value
        else:
            for key, value in self.features['mappings'].items():
                if key in self.features['trb_d'].keys():
                    trb[key] = self.features['trb_d'][key]
                else:
                    if len(value) > 0:
                        if type(value) == list and type(value[0]) != tuple:
                            value=np.array(value)
                    trb[key] = value
            
        with open(f'{self.out_prefix}.trb','wb') as f_out:
            pickle.dump(trb, f_out)
    
    def save_args(self):
        '''
            save args
        '''

        with open(f'{self.out_prefix}_args.json','w') as f_out:
            json.dump(self.args, f_out)

#####################################################################
###################### science is cool ##############################
#####################################################################

# EXTRA SAMPLERS
class cleavage_foldswitch_SAMPLER(SEQDIFF_sampler):
    
    def __init__(self, args=None):
        super().__init__(args)

    def make_dssp_features(self):
        '''
            set up dssp features
        '''

        assert 'child_a_secondary_structure' in self.args.keys() and 'child_b_secondary_structure' in self.args.keys(), \
               f'You are in cleavage triggered foldswitch mode, please specify dssp features for both children'
        
        assert self.args['secondary_structure'] != None, f'You must supply secondary structure features for parent too'
        

        self.features['secondary_structure'] = [self.dssp_dict[x.upper()] for x in self.args['secondary_structure']]
        self.features['a_secondary_structure'] = [self.dssp_dict[x.upper()] for x in self.args['child_a_secondary_structure']]
        self.features['b_secondary_structure'] = [self.dssp_dict[x.upper()] for x in self.args['child_b_secondary_structure']]

        self.features['dssp_feat'] = torch.nn.functional.one_hot(torch.tensor(self.features['secondary_structure']),num_classes=4)
        self.features['a_dssp_feat'] = torch.nn.functional.one_hot(torch.tensor(self.features['a_secondary_structure']),num_classes=4)
        self.features['b_dssp_feat'] = torch.nn.functional.one_hot(torch.tensor(self.features['b_secondary_structure']),num_classes=4)
        
        assert self.features['dssp_feat'].shape[0] == self.features['a_dssp_feat'].shape[0] + self.features['b_dssp_feat'].shape[0],\
                    f'parent and child dssp sizes must match'
        
        self.child_a_len = self.features['a_dssp_feat'].shape[0]
        

    def setup(self, init_model=True):
        '''
            run init model and init features to get everything prepped to go into model
        '''

        # initialize features
        self.feature_init()

        # initialize potential
        if self.args['potentials'] not in ['', None]:
            self.potential_init()
        else:
            self.potential_list = []
            self.use_potentials = False

        # make hostspot features
        self.make_hotspot_features()

        # make dssp features
        self.make_dssp_features()

        # diffuse sequence and mask features
        self.features['seq'], self.features['msa_masked'], \
        self.features['msa_full'], self.features['xyz_t'], self.features['t1d'], \
        self.features['seq_diffused'] = diff_utils.mask_inputs(self.features['seq_hot'],
                                                               self.features['msa_hot'],
                                                               self.features['msa_extra_hot'],
                                                               self.features['xyz_t'],
                                                               self.features['t1d'],
                                                               input_seq_mask=self.features['mask_seq'],
                                                               input_str_mask=self.features['mask_str'],
                                                               input_t1dconf_mask=self.features['blank_mask'],
                                                               diffuser=self.diffuser,
                                                               t=self.max_t,
                                                               MODEL_PARAM=self.MODEL_PARAM,
                                                               hotspots=self.features['hotspot_feat'],
                                                               dssp=self.features['dssp_feat'],
                                                               v2_mode=self.v2_mode)
        
        # move features to device 
        self.features['idx_pdb'] = self.features['idx_pdb'].long().to(self.DEVICE, non_blocking=True) # (B, L)
        self.features['mask_str'] = self.features['mask_str'][None].to(self.DEVICE, non_blocking=True) # (B, L)
        self.features['xyz_t'] = self.features['xyz_t'][None].to(self.DEVICE, non_blocking=True)
        self.features['t1d'] = self.features['t1d'][None].to(self.DEVICE, non_blocking=True)
        self.features['seq'] = self.features['seq'][None].type(torch.float32).to(self.DEVICE, non_blocking=True)
        self.features['msa'] = self.features['msa'].type(torch.float32).to(self.DEVICE, non_blocking=True)
        self.features['msa_masked'] = self.features['msa_masked'][None].type(torch.float32).to(self.DEVICE, non_blocking=True)
        self.features['msa_full'] = self.features['msa_full'][None].type(torch.float32).to(self.DEVICE, non_blocking=True)
        self.ti_dev =  torsion_indices.to(self.DEVICE, non_blocking=True)
        self.ti_flip = torsion_can_flip.to(self.DEVICE, non_blocking=True)
        self.ang_ref = reference_angles.to(self.DEVICE, non_blocking=True)
        self.features['xyz_prev'] = torch.clone(self.features['xyz_t'][0])
        self.features['seq_diffused'] = self.features['seq_diffused'][None].to(self.DEVICE, non_blocking=True)
        
        # make child a and child b features
        self.features['a_seq'] = torch.clone(self.features['seq'][...,:self.child_a_len])
        self.features['b_seq'] = torch.clone(self.features['seq'][...,self.child_a_len:])
        self.features['a_msa'] = torch.clone(self.features['msa'][...,:self.child_a_len])
        self.features['b_msa'] = torch.clone(self.features['msa'][...,self.child_a_len:])
        self.features['a_msa_masked'] = torch.clone(self.features['msa_masked'][:,:,:,:self.child_a_len])
        self.features['b_msa_masked'] = torch.clone(self.features['msa_masked'][:,:,:,self.child_a_len:])
        self.features['a_msa_full'] = torch.clone(self.features['msa_full'][:,:,:,:self.child_a_len])
        self.features['b_msa_full'] = torch.clone(self.features['msa_full'][:,:,:,self.child_a_len:])
        self.features['a_seq_diffused'] = torch.clone(self.features['seq_diffused'][:,:self.child_a_len])
        self.features['b_seq_diffused'] = torch.clone(self.features['seq_diffused'][:,self.child_a_len:])
        self.features['a_idx_pdb'] = torch.clone(self.features['idx_pdb'][:,:self.child_a_len])
        self.features['b_idx_pdb'] = torch.clone(self.features['idx_pdb'][:,:(self.features['idx_pdb'].shape[1]-self.child_a_len)])
        self.features['a_mask_seq'] = torch.clone(self.features['mask_seq'][:,:self.child_a_len])
        self.features['b_mask_seq'] = torch.clone(self.features['mask_seq'][:,self.child_a_len:])
        self.features['a_mask_str'] = torch.clone(self.features['mask_str'][:,:,:self.child_a_len])
        self.features['b_mask_str'] = torch.clone(self.features['mask_str'][:,:,self.child_a_len:])
        self.features['a_xyz_t'] = torch.clone(self.features['xyz_t'][:,:,:self.child_a_len])
        self.features['b_xyz_t'] = torch.clone(self.features['xyz_t'][:,:,self.child_a_len:])
        self.features['a_t1d'] = torch.clone(self.features['t1d'][:,:,:self.child_a_len])
        self.features['b_t1d'] = torch.clone(self.features['t1d'][:,:,self.child_a_len:])
        self.features['a_xyz_prev'] = torch.clone(self.features['xyz_prev'][:,:self.child_a_len])
        self.features['b_xyz_prev'] = torch.clone(self.features['xyz_prev'][:,self.child_a_len:])
        
        # add secondary structure features for children
        self.features['a_t1d'][:,:,:,24:28] = self.features['a_dssp_feat']
        self.features['b_t1d'][:,:,:,24:28] = self.features['b_dssp_feat']
        
        self.features['B'], _, self.features['N'], self.features['L'] = self.features['msa'].shape
        self.features['a_B'], _, self.features['a_N'], self.features['a_L'] = self.features['a_msa'].shape
        self.features['b_B'], _, self.features['b_N'], self.features['b_L'] = self.features['b_msa'].shape
        
        #make t2d
        self.features['t2d'] = xyz_to_t2d(self.features['xyz_t'])
        self.features['a_t2d'] = xyz_to_t2d(self.features['a_xyz_t'])
        self.features['b_t2d'] = xyz_to_t2d(self.features['b_xyz_t'])

        # get alphas
        self.features['alpha'], self.features['alpha_t'] = diff_utils.get_alphas(self.features['t1d'], self.features['xyz_t'],
                                                                                 self.features['B'], self.features['L'],
                                                                                 self.ti_dev, self.ti_flip, self.ang_ref)
        self.features['a_alpha'], self.features['a_alpha_t'] = diff_utils.get_alphas(self.features['a_t1d'], self.features['a_xyz_t'],
                                                                                 self.features['a_B'], self.features['a_L'],
                                                                                 self.ti_dev, self.ti_flip, self.ang_ref)
        self.features['b_alpha'], self.features['b_alpha_t'] = diff_utils.get_alphas(self.features['b_t1d'], self.features['b_xyz_t'],
                                                                                 self.features['b_B'], self.features['b_L'],
                                                                                 self.ti_dev, self.ti_flip, self.ang_ref)

        # processing template coordinates
        self.features['xyz_t'] = get_init_xyz(self.features['xyz_t'])
        self.features['a_xyz_t'] = get_init_xyz(self.features['a_xyz_t'])
        self.features['b_xyz_t'] = get_init_xyz(self.features['b_xyz_t'])
        self.features['xyz_prev'] = get_init_xyz(self.features['xyz_prev'][:,None]).reshape(self.features['B'], self.features['L'], 27, 3)
        self.features['a_xyz_prev'] = get_init_xyz(self.features['a_xyz_prev'][:,None]).reshape(self.features['a_B'], self.features['a_L'], 27, 3)
        self.features['b_xyz_prev'] = get_init_xyz(self.features['b_xyz_prev'][:,None]).reshape(self.features['b_B'], self.features['b_L'], 27, 3)

        # initialize extra features to none
        self.features['xyz'] = None
        self.features['a_xyz'] = None
        self.features['b_xyz'] = None
        self.features['pred_lddt'] = None
        self.features['a_pred_lddt'] = None
        self.features['b_pred_lddt'] = None
        self.features['logit_s'] = None
        self.features['a_logit_s'] = None
        self.features['b_logit_s'] = None
        self.features['logit_aa_s'] = None
        self.features['a_logit_aa_s'] = None
        self.features['b_logit_aa_s'] = None
        self.features['best_plddt'] = 0
        self.features['a_best_plddt'] = 0
        self.features['b_best_plddt'] = 0
        self.features['best_pred_lddt'] = torch.zeros_like(self.features['mask_str'])[0].float()
        self.features['a_best_pred_lddt'] = torch.zeros_like(self.features['a_mask_str'])[0].float()
        self.features['b_best_pred_lddt'] = torch.zeros_like(self.features['b_mask_str'])[0].float()
        self.features['msa_prev'] = None
        self.features['a_msa_prev'] = None
        self.features['b_msa_prev'] = None
        self.features['pair_prev'] = None
        self.features['a_pair_prev'] = None
        self.features['b_pair_prev'] = None
        self.features['state_prev'] = None
        self.features['a_state_prev'] = None
        self.features['b_state_prev'] = None
    
    def predict_x(self):
        '''
            take step using X_t-1 features to predict Xo
        '''
        for prefix in ['','a_','b_']:
            
            self.features[f'{prefix}seq'], \
            self.features[f'{prefix}xyz'], \
            self.features[f'{prefix}pred_lddt'], \
            self.features[f'{prefix}logit_s'], \
            self.features[f'{prefix}logit_aa_s'], \
            self.features[f'{prefix}alpha'], \
            self.features[f'{prefix}msa_prev'], \
            self.features[f'{prefix}pair_prev'], \
            self.features[f'{prefix}state_prev'] \
            = diff_utils.take_step_nostate(self.model,
            self.features[f'{prefix}msa_masked'],
            self.features[f'{prefix}msa_full'],
            self.features[f'{prefix}seq'],
            self.features[f'{prefix}t1d'],
            self.features[f'{prefix}t2d'],
            self.features[f'{prefix}idx_pdb'],
            self.args['n_cycle'],
            self.features[f'{prefix}xyz_prev'],
            self.features[f'{prefix}alpha'],
            self.features[f'{prefix}xyz_t'],
            self.features[f'{prefix}alpha_t'],
            self.features[f'{prefix}seq_diffused'],
            self.features[f'{prefix}msa_prev'],
            self.features[f'{prefix}pair_prev'],
            self.features[f'{prefix}state_prev'])
            
    def self_condition_seq(self):
        '''
            get previous logits and set at t1d template
        '''
        self.features['t1d'][:,:,:,:21] = self.features['seq_out'][:,:21]
    
    def noise_x(self):
        '''
            get X_t-1 from predicted Xo
        '''
        # sample x_t-1
        self.features['post_mean'] = self.diffuser.q_sample(self.features['seq_out'], self.t, DEVICE=self.DEVICE)

        if self.features['sym'] > 1:
            self.features['post_mean'] = self.symmetrize_seq(self.features['post_mean'])

        # update seq and masks
        self.features['seq_diffused'][0,~self.features['mask_seq'][0],:21] = self.features['post_mean'][~self.features['mask_seq'][0],...]
        self.features['seq_diffused'][0,:,21] = 0.0

        # did not know we were clamping seq
        self.features['seq_diffused'] = torch.clamp(self.features['seq_diffused'], min=-3, max=3)

        # match other features to seq diffused
        self.features['seq'] = torch.argmax(self.features['seq_diffused'], dim=-1)[None]
        self.features['msa_masked'][:,:,:,:,:22] = self.features['seq_diffused']
        self.features['msa_masked'][:,:,:,:,22:44] = self.features['seq_diffused']
        self.features['msa_full'][:,:,:,:,:22] = self.features['seq_diffused']
        self.features['t1d'][:1,:,:,22] = 1-int(self.t)/self.args['T']
        
        # get clones for children
        self.features['a_seq'] = torch.clone(self.features['seq'][...,:self.child_a_len])
        self.features['b_seq'] = torch.clone(self.features['seq'][...,self.child_a_len:])
        self.features['a_msa'] = torch.clone(self.features['msa'][...,:self.child_a_len])
        self.features['b_msa'] = torch.clone(self.features['msa'][...,self.child_a_len:])
        self.features['a_msa_masked'] = torch.clone(self.features['msa_masked'][:,:,:,:self.child_a_len])
        self.features['b_msa_masked'] = torch.clone(self.features['msa_masked'][:,:,:,self.child_a_len:])
        self.features['a_msa_full'] = torch.clone(self.features['msa_full'][:,:,:,:self.child_a_len])
        self.features['b_msa_full'] = torch.clone(self.features['msa_full'][:,:,:,self.child_a_len:])
        self.features['a_seq_diffused'] = torch.clone(self.features['seq_diffused'][:,:self.child_a_len])
        self.features['b_seq_diffused'] = torch.clone(self.features['seq_diffused'][:,self.child_a_len:])
        self.features['a_t1d'][:,:,:,:21] = torch.clone(self.features['t1d'][:,:,:self.child_a_len,:21])
        self.features['b_t1d'][:,:,:,:21] = torch.clone(self.features['t1d'][:,:,self.child_a_len:,:21])
        self.features['a_t1d'][:,:,:,22] = 1-int(self.t)/self.args['T']
        self.features['b_t1d'][:,:,:,22] = 1-int(self.t)/self.args['T']
        
    def generate_sample(self):
        '''
            sample from the model 
            
            this function runs the full sampling loop
        '''
        # setup example
        self.setup()

        # start time
        self.start_time = time.time()

        # set up dictionary to save at each step in trajectory
        self.trajectory = {}

        # set out prefix
        self.out_prefix = self.args['out']+f'_{self.design_num:06}'
        print(f'Generating sample {self.design_num:06} ...')

        # main sampling loop
        for j in range(self.max_t):
            self.t = torch.tensor(self.max_t-j-1).to(self.DEVICE)

            # run features through the model to get X_o prediction
            self.predict_x()

            # save step
            if self.args['save_all_steps']:
                self.save_step()

            # get seq out
            self.features['seq_out'] = torch.permute(self.features['logit_aa_s'][0], (1,0))
            self.features['a_seq_out'] = torch.permute(self.features['a_logit_aa_s'][0], (1,0))
            self.features['b_seq_out'] = torch.permute(self.features['b_logit_aa_s'][0], (1,0))
            
            # mix seq out for update
            ab_seq_out = torch.cat([self.features['a_seq_out'], self.features['b_seq_out']], dim=0)
            self.features['seq_out'] = (self.features['seq_out'] * (1-self.args['mixing'])) + (ab_seq_out * self.args['mixing'])
            
            self.features['parent_plddt'] = torch.clone(self.features['pred_lddt'][~self.features['mask_seq']]).mean().item()
            self.features['child_a_plddt'] = torch.clone(self.features['a_pred_lddt'][~self.features['a_mask_seq']]).mean().item()
            self.features['child_b_plddt'] = torch.clone(self.features['b_pred_lddt'][~self.features['b_mask_seq']]).mean().item()
            self.features['mix_plddt'] = self.features['parent_plddt']*(1-self.args['mixing']) + \
                        (((self.features['child_a_plddt']+self.features['child_b_plddt'])/2)*self.args['mixing'])
            
            # save best seq
            if self.features['mix_plddt'] > self.features['best_plddt']:
                self.features['best_seq'] = torch.argmax(torch.clone(self.features['seq_out']), dim=-1)
                self.features['a_best_seq'] = torch.argmax(torch.clone(self.features['a_seq_out']), dim=-1)
                self.features['b_best_seq'] = torch.argmax(torch.clone(self.features['b_seq_out']), dim=-1)
                self.features['best_pred_lddt'] = torch.clone(self.features['pred_lddt'])
                self.features['a_best_pred_lddt'] = torch.clone(self.features['a_pred_lddt'])
                self.features['b_best_pred_lddt'] = torch.clone(self.features['b_pred_lddt'])
                self.features['best_xyz'] = torch.clone(self.features['xyz'])
                self.features['a_best_xyz'] = torch.clone(self.features['a_xyz'])
                self.features['b_best_xyz'] = torch.clone(self.features['b_xyz'])
                self.features['best_plddt'] = copy.deepcopy(self.features['mix_plddt'])
            
            
            # self condition on sequence
            self.self_condition_seq()

            # self condition on structure
            if self.args['scheduled_str_cond']:
                self.self_condition_str_scheduled()
            if self.args['struc_cond_sc']:
                self.self_condition_str()

            # sequence alterations
            if self.args['softmax_seqout']:
                self.features['seq_out'] = torch.softmax(self.features['seq_out'],dim=-1)*2-1
            if self.args['clamp_seqout']:
                self.features['seq_out'] = torch.clamp(self.features['seq_out'],
                                                       min=-((1/self.diffuser.alphas_cumprod[t])*0.25+5),
                                                       max=((1/self.diffuser.alphas_cumprod[t])*0.25+5))

            # apply potentials
            if self.use_potentials:
                self.apply_potentials()

            # noise to X_t-1
            if self.t != 0:
                self.noise_x()
            
            print(f'MIX PLDDT: {self.features["mix_plddt"]:.03f}     BEST MIX PLDDT: {self.features["best_plddt"]:.03f}')
            
            print('PARENT: '+''.join([self.conversion[i] for i in torch.argmax(self.features['seq_out'],dim=-1)]))
            print ("    TIMESTEP [%02d/%02d]   |   current PLDDT: %.4f   <<  >>   best PLDDT: %.4f"%(
                    self.t+1, self.args['T'], self.features['pred_lddt'][~self.features['mask_seq']].mean().item(),
                    self.features['best_pred_lddt'][~self.features['mask_seq']].mean().item()))
            
            print('CHILD A: '+''.join([self.conversion[i] for i in torch.argmax(self.features['a_seq_out'],dim=-1)]))
            print ("    TIMESTEP [%02d/%02d]   |   current PLDDT: %.4f   <<  >>   best PLDDT: %.4f"%(
                    self.t+1, self.args['T'], self.features['a_pred_lddt'][~self.features['a_mask_seq']].mean().item(),
                    self.features['a_best_pred_lddt'][~self.features['a_mask_seq']].mean().item()))
            
            print('CHILD B: '+''.join([self.conversion[i] for i in torch.argmax(self.features['b_seq_out'],dim=-1)]))
            print ("    TIMESTEP [%02d/%02d]   |   current PLDDT: %.4f   <<  >>   best PLDDT: %.4f"%(
                    self.t+1, self.args['T'], self.features['b_pred_lddt'][~self.features['b_mask_seq']].mean().item(),
                    self.features['b_best_pred_lddt'][~self.features['b_mask_seq']].mean().item()))
            
            
        # record time
        self.delta_time = time.time() - self.start_time

        # save outputs
        self.save_outputs()

        # increment design num
        self.design_num += 1

        print(f'Finished design {self.out_prefix} in {self.delta_time/60:.2f} minutes.')
    
    def save_outputs(self):
        '''
            save the outputs from the model
        '''
        # save trajectory
        if self.args['save_all_steps']:
            fname = f'{self.out_prefix}_trajectory.pt'
            torch.save(self.trajectory, fname)

        # get items from best plddt step
        if self.args['save_best_plddt']:
            self.features['seq'] = torch.clone(self.features['best_seq'])
            self.features['a_seq'] = torch.clone(self.features['a_best_seq'])
            self.features['b_seq'] = torch.clone(self.features['b_best_seq'])
            self.features['pred_lddt'] = torch.clone(self.features['best_pred_lddt'])
            self.features['a_pred_lddt'] = torch.clone(self.features['a_best_pred_lddt'])
            self.features['b_pred_lddt'] = torch.clone(self.features['b_best_pred_lddt'])
            self.features['xyz'] = torch.clone(self.features['best_xyz'])
            self.features['a_xyz'] = torch.clone(self.features['a_best_xyz'])
            self.features['b_xyz'] = torch.clone(self.features['b_best_xyz'])
            
        if self.args['match_seqs']:
            self.features['a_seq'] = torch.clone(self.features['seq'][:self.child_a_len])
            self.features['b_seq'] = torch.clone(self.features['seq'][self.child_a_len:])
            
        
        # get chain IDs
        if (self.args['sampling_temp'] == 1.0 and self.args['trb'] == None) or (self.args['sequence'] not in ['',None]):
            chain_ids = [i[0] for i in self.features['pdb_idx']]
        elif self.args['dump_pdb']:
            chain_ids = [i[0] for i in self.features['parsed_pdb']['pdb_idx']]
        
        a_chain_ids = chain_ids[:self.child_a_len]
        b_chain_ids = chain_ids[self.child_a_len:]
        
        # write output pdb
        if len(self.features['seq'].shape) == 2:
            self.features['seq'] = self.features['seq'].squeeze()
            self.features['a_seq'] = self.features['a_seq'].squeeze()
            self.features['b_seq'] = self.features['b_seq'].squeeze()
            
        fname = self.out_prefix + '_parent.pdb'
        write_pdb(fname,
                  self.features['seq'].type(torch.int64),
                  self.features['xyz'].squeeze(),
                  Bfacts=self.features['pred_lddt'].squeeze(),
                  chains=chain_ids)
        
        a_fname = self.out_prefix + '_a_child.pdb'
        write_pdb(a_fname,
                  self.features['a_seq'].type(torch.int64),
                  self.features['a_xyz'].squeeze(),
                  Bfacts=self.features['a_pred_lddt'].squeeze(),
                  chains=a_chain_ids)
        
        b_fname = self.out_prefix + '_b_child.pdb'
        write_pdb(b_fname,
                  self.features['b_seq'].type(torch.int64),
                  self.features['b_xyz'].squeeze(),
                  Bfacts=self.features['b_pred_lddt'].squeeze(),
                  chains=b_chain_ids)

        if self.args['dump_trb']:
            self.save_trb()

        if self.args['save_args']:
            self.save_args()
