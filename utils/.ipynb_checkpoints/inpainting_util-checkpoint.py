import math
import os
import csv
import random
import torch
from torch.utils import data
import numpy as np
from dateutil import parser
import contigs
from util import *
from kinematics import *
import pandas as pd
import sys
import torch.nn as nn
from icecream import ic
def write_pdb(filename, seq, atoms, Bfacts=None, prefix=None, chains=None):
        L = len(seq)
        ctr = 1 
        seq = seq.long()
        with open(filename, 'w+') as f:
            for i,s in enumerate(seq):
                if chains is None:
                    chain='A'
                else:
                    chain=chains[i]

                if (len(atoms.shape)==2):
                    f.write ("%-6s%5s %4s %3s %s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f\n"%(
                            "ATOM", ctr, " CA ", util.num2aa[s], 
                            chain, i+1, atoms[i,0], atoms[i,1], atoms[i,2],
                            1.0, Bfacts[i] ) ) 
                    ctr += 1
    
                elif atoms.shape[1]==3:
                    for j,atm_j in enumerate((" N  "," CA "," C  ")):
                        f.write ("%-6s%5s %4s %3s %s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f\n"%(
                                "ATOM", ctr, atm_j, num2aa[s], 
                                chain, i+1, atoms[i,j,0], atoms[i,j,1], atoms[i,j,2],
                                1.0, Bfacts[i] ) ) 
                        ctr += 1    
                else:
                    atms = aa2long[s]
                    for j,atm_j in enumerate(atms):
                        if (atm_j is not None):
                            f.write ("%-6s%5s %4s %3s %s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f\n"%(
                                "ATOM", ctr, atm_j, num2aa[s], 
                                chain, i+1, atoms[i,j,0], atoms[i,j,1], atoms[i,j,2],
                                1.0, Bfacts[i] ) ) 
                            ctr += 1

def preprocess(xyz_t, t1d, DEVICE, masks_1d, ti_dev=None, ti_flip=None, ang_ref=None):

      B, _, L, _, _ = xyz_t.shape

      seq_tmp = t1d[...,:-1].argmax(dim=-1).reshape(-1,L).to(DEVICE, non_blocking=True)
      alpha, _, alpha_mask,_ = get_torsions(xyz_t.reshape(-1,L,27,3), seq_tmp, ti_dev, ti_flip, ang_ref)
      alpha_mask = torch.logical_and(alpha_mask, ~torch.isnan(alpha[...,0]))
      alpha[torch.isnan(alpha)] = 0.0
      alpha = alpha.reshape(B,-1,L,10,2)
      alpha_mask = alpha_mask.reshape(B,-1,L,10,1)
      alpha_t = torch.cat((alpha, alpha_mask), dim=-1).reshape(B,-1,L,30)
      #t1d = torch.cat((t1d, chis.reshape(B,-1,L,30)), dim=-1)
      xyz_t = get_init_xyz(xyz_t)
      xyz_prev = xyz_t[:,0]
      state = t1d[:,0]
      alpha = alpha[:,0]
      t2d=xyz_to_t2d(xyz_t)
      return (t2d, alpha, alpha_mask, alpha_t, t1d, xyz_t, xyz_prev, state)

def TemplFeaturizeFixbb(seq, conf_1d=None):
    """  
    Template 1D featurizer for fixed BB examples :
    Parameters:
        seq (torch.tensor, required): Integer sequence 
        conf_1d (torch.tensor, optional): Precalcualted confidence tensor
    """
    L = seq.shape[-1]
    t1d  = torch.nn.functional.one_hot(seq, num_classes=21) # one hot sequence 
    if conf_1d is None:
        conf = torch.ones_like(seq)[...,None]
    else:
        conf = conf_1d[:,None]
    t1d = torch.cat((t1d, conf), dim=-1)
    return t1d  

def MSAFeaturize_fixbb(msa, params):
    '''
    Input: full msa information
    Output: Single sequence, with some percentage of amino acids mutated (but no resides 'masked')
    
    This is modified from autofold2, to remove mutations of the single sequence
    '''
    N, L = msa.shape
    # raw MSA profile
    raw_profile = torch.nn.functional.one_hot(msa, num_classes=22)
    raw_profile = raw_profile.float().mean(dim=0)

    b_seq = list()
    b_msa_clust = list()
    b_msa_seed = list()
    b_msa_extra = list()
    b_mask_pos = list()
    for i_cycle in range(params['MAXCYCLE']):
        assert torch.max(msa) < 22
        msa_onehot = torch.nn.functional.one_hot(msa[:1],num_classes=22)
        msa_fakeprofile_onehot = torch.nn.functional.one_hot(msa[:1],num_classes=26) #add the extra two indel planes, which will be set to zero
        msa_full_onehot = torch.cat((msa_onehot, msa_fakeprofile_onehot), dim=-1)

        #make fake msa_extra
        msa_extra_onehot = torch.nn.functional.one_hot(msa[:1],num_classes=25)

        #make fake msa_clust and mask_pos
        msa_clust = msa[:1]
        mask_pos = torch.full_like(msa_clust, 1).bool()
        b_seq.append(msa[0].clone())
        b_msa_seed.append(msa_full_onehot[:1].clone()) #masked single sequence onehot (nb no mask so just single sequence onehot)
        b_msa_extra.append(msa_extra_onehot[:1].clone()) #masked single sequence onehot (nb no mask so just single sequence onehot)
        b_msa_clust.append(msa_clust[:1].clone()) #unmasked original single sequence 
        b_mask_pos.append(mask_pos[:1].clone()) #mask positions in single sequence (all zeros)

    b_seq = torch.stack(b_seq)
    b_msa_clust = torch.stack(b_msa_clust)
    b_msa_seed = torch.stack(b_msa_seed)
    b_msa_extra = torch.stack(b_msa_extra)
    b_mask_pos = torch.stack(b_mask_pos)

    return b_seq, b_msa_clust, b_msa_seed, b_msa_extra, b_mask_pos

def MSAFeaturize(msa, params):
    '''
    Input: full msa information
    Output: Single sequence, with some percentage of amino acids mutated (but no resides 'masked')
    
    This is modified from autofold2, to remove mutations of the single sequence
    '''
    N, L = msa.shape
    # raw MSA profile
    raw_profile = torch.nn.functional.one_hot(msa, num_classes=22)
    raw_profile = raw_profile.float().mean(dim=0)

    b_seq = list()
    b_msa_clust = list()
    b_msa_seed = list()
    b_msa_extra = list()
    b_mask_pos = list()
    for i_cycle in range(params['MAXCYCLE']):
        assert torch.max(msa) < 22
        msa_onehot = torch.nn.functional.one_hot(msa,num_classes=22)
        msa_fakeprofile_onehot = torch.nn.functional.one_hot(msa,num_classes=26) #add the extra two indel planes, which will be set to zero
        msa_full_onehot = torch.cat((msa_onehot, msa_fakeprofile_onehot), dim=-1)

        #make fake msa_extra
        msa_extra_onehot = torch.nn.functional.one_hot(msa,num_classes=25)

        #make fake msa_clust and mask_pos
        msa_clust = msa
        mask_pos = torch.full_like(msa_clust, 1).bool()
        b_seq.append(msa[0].clone())
        b_msa_seed.append(msa_full_onehot.clone()) #masked single sequence onehot (nb no mask so just single sequence onehot)
        b_msa_extra.append(msa_extra_onehot.clone()) #masked single sequence onehot (nb no mask so just single sequence onehot)
        b_msa_clust.append(msa_clust.clone()) #unmasked original single sequence 
        b_mask_pos.append(mask_pos.clone()) #mask positions in single sequence (all zeros)

    b_seq = torch.stack(b_seq)
    b_msa_clust = torch.stack(b_msa_clust)
    b_msa_seed = torch.stack(b_msa_seed)
    b_msa_extra = torch.stack(b_msa_extra)
    b_mask_pos = torch.stack(b_mask_pos)

    return b_seq, b_msa_clust, b_msa_seed, b_msa_extra, b_mask_pos

def mask_inputs(seq, msa_masked, msa_full, xyz_t, t1d, input_seq_mask=None, input_str_mask=None, input_t1dconf_mask=None, loss_seq_mask=None, loss_str_mask=None):
    """
    Parameters:
        seq (torch.tensor, required): (B,I,L) integer sequence 
        msa_masked (torch.tensor, required): (B,I,N_short,L,46)
        msa_full  (torch,.tensor, required): (B,I,N_long,L,23)
        
        xyz_t (torch,tensor): (B,T,L,14,3) template crds BEFORE they go into get_init_xyz 
        
        t1d (torch.tensor, required): (B,I,L,22) this is the t1d before tacking on the chi angles 
        
        str_mask_1D (torch.tensor, required): Shape (L) rank 1 tensor where structure is masked at False positions 
        seq_mask_1D (torch.tensor, required): Shape (L) rank 1 tensor where seq is masked at False positions 
    """

    ###########
    B,_,_ = seq.shape
    assert B == 1, 'batch sizes > 1 not supported'
    seq_mask = input_seq_mask[0]
    seq[:,:,~seq_mask] = 21 # mask token categorical value

    ### msa_masked ###
    ################## 
    msa_masked[:,:,:,~seq_mask,:20] = 0
    msa_masked[:,:,:,~seq_mask,20]  = 0
    msa_masked[:,:,:,~seq_mask,21]  = 1     # set to the unkown char
    
    # index 44/45 is insertion/deletion
    # index 43 is the unknown token
    # index 42 is the masked token 
    msa_masked[:,:,:,~seq_mask,22:42] = 0
    msa_masked[:,:,:,~seq_mask,43] = 1 
    msa_masked[:,:,:,~seq_mask,42] = 0

    # insertion/deletion stuff 
    msa_masked[:,:,:,~seq_mask,44:] = 0

    ### msa_full ### 
    ################
    msa_full[:,:,:,~seq_mask,:20] = 0
    msa_full[:,:,:,~seq_mask,21]  = 1
    msa_full[:,:,:,~seq_mask,20]  = 0 
    msa_full[:,:,:,~seq_mask,-1]  = 0   #NOTE: double check this is insertions/deletions and 0 makes sense 

    ### t1d ###
    ########### 
    # NOTE: Not adjusting t1d last dim (confidence) from sequence mask
    t1d[:,:,~seq_mask,:20] = 0 
    t1d[:,:,~seq_mask,20]  = 1 # unknown

    t1d[:,:,:,21] *= input_t1dconf_mask

    #JG added in here to make sure everything fits
    print('expanding t1d to 24 dims')
    
    t1d = torch.cat((t1d, torch.zeros((t1d.shape[0],t1d.shape[1],t1d.shape[2],2)).float()), -1).to(seq.device)

    xyz_t[:,:,~seq_mask,3:,:] = float('nan')

    # Structure masking
    str_mask = input_str_mask[0]
    xyz_t[:,:,~str_mask,:,:] = float('nan')

    return seq, msa_masked, msa_full, xyz_t, t1d
    

###########################################################
#Functions for randomly translating/rotation input residues
###########################################################

def get_translated_coords(args):
    '''
    Parses args.res_translate
    '''
    #get positions to translate
    res_translate = []
    for res in args.res_translate.split(":"):
        temp_str = []
        for i in res.split(','):
            temp_str.append(i)
        if temp_str[-1][0].isalpha() is True:
            temp_str.append(2.0) #set default distance
        for i in temp_str[:-1]:
            if '-' in i:
                start = int(i.split('-')[0][1:])
                while start <= int(i.split('-')[1]):
                    res_translate.append((i.split('-')[0][0] + str(start),float(temp_str[-1])))
                    start += 1
            else:
                res_translate.append((i, float(temp_str[-1])))
        start = 0
    
    output = []
    for i in res_translate:
        temp = (i[0], i[1], start)
        output.append(temp)
        start += 1

    return output

def get_tied_translated_coords(args, untied_translate=None):
    '''
    Parses args.tie_translate
    '''
    #pdb_idx = list(parsed_pdb['idx'])
    #xyz = parsed_pdb['xyz']
    #get positions to translate
    res_translate = []
    block = 0
    for res in args.tie_translate.split(":"):
        temp_str = []
        for i in res.split(','):
            temp_str.append(i)
        if temp_str[-1][0].isalpha() is True:
            temp_str.append(2.0) #set default distance
        for i in temp_str[:-1]:
            if '-' in i:
                start = int(i.split('-')[0][1:])
                while start <= int(i.split('-')[1]):
                    res_translate.append((i.split('-')[0][0] + str(start),float(temp_str[-1]), block))
                    start += 1
            else:
                res_translate.append((i, float(temp_str[-1]), block))
        block += 1
    
    #sanity check
    if untied_translate != None:
        checker = [i[0] for i in res_translate]
        untied_check = [i[0] for i in untied_translate]
        for i in checker:
            if i in untied_check:
                print(f'WARNING: residue {i} is specified both in --res_translate and --tie_translate. Residue {i} will be ignored in --res_translate, and instead only moved in a tied block (--tie_translate)')
        
        final_output = res_translate
        for i in untied_translate:
            if i[0] not in checker:
                final_output.append((i[0],i[1],i[2] + block + 1))
    else:
        final_output = res_translate
    
    return final_output

 

def translate_coords(parsed_pdb, res_translate):
    '''
    Takes parsed list in format [(chain_residue,distance,tieing_block)] and randomly translates residues accordingly.
    '''

    pdb_idx = parsed_pdb['pdb_idx']
    xyz = np.copy(parsed_pdb['xyz'])
    translated_coord_dict = {}
    #get number of blocks
    temp = [int(i[2]) for i in res_translate]
    blocks = np.max(temp)

    for block in range(blocks + 1):
        init_dist = 1.01
        while init_dist > 1: #gives equal probability to any direction (as keeps going until init_dist is within unit circle)
            x = random.uniform(-1,1)
            y = random.uniform(-1,1)
            z = random.uniform(-1,1)
            init_dist = np.sqrt(x**2 + y**2 + z**2)
        x=x/init_dist
        y=y/init_dist
        z=z/init_dist
        translate_dist = random.uniform(0,1) #now choose distance (as proportion of maximum) that coordinates will be translated
        for res in res_translate:
            if res[2] == block:
                res_idx = pdb_idx.index((res[0][0],int(res[0][1:])))
                original_coords = np.copy(xyz[res_idx,:,:])
                for i in range(14):
                    if parsed_pdb['mask'][res_idx, i]:
                        xyz[res_idx,i,0] += np.float32(x * translate_dist * float(res[1]))
                        xyz[res_idx,i,1] += np.float32(y * translate_dist * float(res[1]))
                        xyz[res_idx,i,2] += np.float32(z * translate_dist * float(res[1]))
                translated_coords = xyz[res_idx,:,:]
                translated_coord_dict[res[0]] = (original_coords.tolist(), translated_coords.tolist())
         
    return xyz[:,:,:], translated_coord_dict

def parse_block_rotate(args):
    block_translate = []
    block = 0
    for res in args.block_rotate.split(":"):
        temp_str = []
        for i in res.split(','):
            temp_str.append(i)
        if temp_str[-1][0].isalpha() is True:
            temp_str.append(10) #set default angle to 10 degrees
        for i in temp_str[:-1]:
            if '-' in i:
                start = int(i.split('-')[0][1:])
                while start <= int(i.split('-')[1]):
                    block_translate.append((i.split('-')[0][0] + str(start),float(temp_str[-1]), block))
                    start += 1
            else:
                block_translate.append((i, float(temp_str[-1]), block))
        block += 1
    return block_translate

def rotate_block(xyz, block_rotate,pdb_index):
    rotated_coord_dict = {}
    #get number of blocks
    temp = [int(i[2]) for i in block_rotate]
    blocks = np.max(temp)
    for block in range(blocks + 1):
        idxs = [pdb_index.index((i[0][0],int(i[0][1:]))) for i in block_rotate if i[2] == block]
        angle = [i[1] for i in block_rotate if i[2] == block][0]
        block_xyz = xyz[idxs,:,:]
        com = [float(torch.mean(block_xyz[:,:,i])) for i in range(3)]
        origin_xyz = np.copy(block_xyz)
        for i in range(np.shape(origin_xyz)[0]):
            for j in range(14):
                origin_xyz[i,j] = origin_xyz[i,j] - com
        rotated_xyz = rigid_rotate(origin_xyz,angle,angle,angle)
        recovered_xyz = np.copy(rotated_xyz)
        for i in range(np.shape(origin_xyz)[0]):
            for j in range(14):
                recovered_xyz[i,j] = rotated_xyz[i,j] + com
        recovered_xyz=torch.tensor(recovered_xyz)
        rotated_coord_dict[f'rotated_block_{block}_original'] = block_xyz
        rotated_coord_dict[f'rotated_block_{block}_rotated'] = recovered_xyz
        xyz_out = torch.clone(xyz)
        for i in range(len(idxs)):
            xyz_out[idxs[i]] = recovered_xyz[i]
    return xyz_out,rotated_coord_dict

def rigid_rotate(xyz,a=180,b=180,c=180):
    #TODO fix this to make it truly uniform
    a=(a/180)*math.pi
    b=(b/180)*math.pi
    c=(c/180)*math.pi
    alpha = random.uniform(-a, a)
    beta = random.uniform(-b, b)
    gamma = random.uniform(-c, c)
    rotated = []
    for i in range(np.shape(xyz)[0]):
        for j in range(14):
            try:
                x = xyz[i,j,0]
                y = xyz[i,j,1]
                z = xyz[i,j,2]
                x2 = x*math.cos(alpha) - y*math.sin(alpha)
                y2 = x*math.sin(alpha) + y*math.cos(alpha)
                x3 = x2*math.cos(beta) - z*math.sin(beta)
                z2 = x2*math.sin(beta) + z*math.cos(beta)
                y3 = y2*math.cos(gamma) - z2*math.sin(gamma)
                z3 = y2*math.sin(gamma) + z2*math.cos(gamma)
                rotated.append([x3,y3,z3])
            except:
                rotated.append([float('nan'),float('nan'),float('nan')])
    rotated=np.array(rotated)
    rotated=np.reshape(rotated, [np.shape(xyz)[0],14,3])
    
    return rotated


######## from old pred_util.py 
def find_contigs(mask):
    """
    Find contiguous regions in a mask that are True with no False in between

    Parameters:
        mask (torch.tensor or np.array, required): 1D boolean array 

    Returns:
        contigs (list): List of tuples, each tuple containing the beginning and the  
    """
    assert len(mask.shape) == 1 # 1D tensor of bools 
    
    contigs = []
    found_contig = False 
    for i,b in enumerate(mask):
        
        
        if b and not found_contig:   # found the beginning of a contig
            contig = [i]
            found_contig = True 
        
        elif b and found_contig:     # currently have contig, continuing it 
            pass 
        
        elif not b and found_contig: # found the end, record previous index as end, reset indicator  
            contig.append(i)
            found_contig = False 
            contigs.append(tuple(contig))
        
        else:                        # currently don't have a contig, and didn't find one 
            pass 
    
    
    # fence post bug - check if the very last entry was True and we didn't get to finish 
    if b:
        contig.append(i+1)
        found_contig = False 
        contigs.append(tuple(contig))
        
    return contigs


def reindex_chains(pdb_idx):
    """
    Given a list of (chain, index) tuples, and the indices where chains break, create a reordered indexing 

    Parameters:
        
        pdb_idx (list, required): List of tuples (chainID, index) 

        breaks (list, required): List of indices where chains begin 
    """

    new_breaks, new_idx = [],[]
    current_chain = None

    chain_and_idx_to_torch = {}

    for i,T in enumerate(pdb_idx):

        chain, idx = T

        if chain != current_chain:
            new_breaks.append(i)
            current_chain = chain 
            
            # create new space for chain id listings 
            chain_and_idx_to_torch[chain] = {}
        
        # map original pdb (chain, idx) pair to index in tensor 
        chain_and_idx_to_torch[chain][idx] = i
        
        # append tensor index to list 
        new_idx.append(i)
    
    new_idx = np.array(new_idx)
    # now we have ordered list and know where the chainbreaks are in the new order 
    num_additions = 0
    for i in new_breaks[1:]: # skip the first trivial one
        new_idx[np.where(new_idx==(i+ num_additions*500))[0][0]:] += 500
        num_additions += 1
    
    return new_idx, chain_and_idx_to_torch,new_breaks[1:]

class ObjectView(object):
    '''
    Easy wrapper to access dictionary values with "dot" notiation instead
    '''
    def __init__(self, d):
        self.__dict__ = d

def split_templates(xyz_t, t1d, multi_templates,mappings,multi_tmpl_conf=None):
    templates = multi_templates.split(":")
    if multi_tmpl_conf is not None:
        multi_tmpl_conf = [float(i) for i in multi_tmpl_conf.split(",")]
        assert len(templates) == len(multi_tmpl_conf), "Number of templates must equal number of confidences specified in --multi_tmpl_conf flag"
    for idx, template in enumerate(templates):
        parts = template.split(",")
        template_mask = torch.zeros(xyz_t.shape[2]).bool()
        for part in parts:
            start = int(part.split("-")[0][1:])
            end = int(part.split("-")[1]) + 1
            chain = part[0]
            for i in range(start, end):
                try:
                    ref_pos = mappings['complex_con_ref_pdb_idx'].index((chain, i))
                    hal_pos_0 = mappings['complex_con_hal_idx0'][ref_pos]
                except:
                    ref_pos = mappings['con_ref_pdb_idx'].index((chain, i))
                    hal_pos_0 = mappings['con_hal_idx0'][ref_pos]
                template_mask[hal_pos_0] = True

        xyz_t_temp = torch.clone(xyz_t)
        xyz_t_temp[:,:,~template_mask,:,:] = float('nan')
        t1d_temp = torch.clone(t1d)
        t1d_temp[:,:,~template_mask,:20] =0
        t1d_temp[:,:,~template_mask,20] = 1
        if multi_tmpl_conf is not None:
            t1d_temp[:,:,template_mask,21] = multi_tmpl_conf[idx]
        if idx != 0:
            xyz_t_out = torch.cat((xyz_t_out, xyz_t_temp),dim=1)
            t1d_out = torch.cat((t1d_out, t1d_temp),dim=1)
        else:
            xyz_t_out = xyz_t_temp
            t1d_out = t1d_temp
    return xyz_t_out, t1d_out


class ContigMap():
    '''
    New class for doing mapping.
    Supports multichain or multiple crops from a single receptor chain.
    Also supports indexing jump (+200) or not, based on contig input.
    Default chain outputs are inpainted chains as A (and B, C etc if multiple chains), and all fragments of receptor chain on the next one (generally B)
    Output chains can be specified. Sequence must be the same number of elements as in contig string
    '''
    def __init__(self, parsed_pdb, contigs=None, inpaint_seq=None, inpaint_str=None, length=None, ref_idx=None, hal_idx=None, idx_rf=None, inpaint_seq_tensor=None, inpaint_str_tensor=None, topo=False):
        #sanity checks
        if contigs is None and ref_idx is None:
            sys.exit("Must either specify a contig string or precise mapping")
        if idx_rf is not None or hal_idx is not None or ref_idx is not None:
            if idx_rf is None or hal_idx is None or ref_idx is None:
                sys.exit("If you're specifying specific contig mappings, the reference and output positions must be specified, AND the indexing for RoseTTAFold (idx_rf)")
        
        self.chain_order='ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        if length is not None:
            if '-' not in length:
                self.length = [int(length),int(length)+1]
            else:
                self.length = [int(length.split("-")[0]),int(length.split("-")[1])+1]
        else:
            self.length = None
        self.ref_idx = ref_idx
        self.hal_idx=hal_idx
        self.idx_rf=idx_rf
        self.inpaint_seq = ','.join(inpaint_seq).split(",") if inpaint_seq is not None else None
        self.inpaint_str = ','.join(inpaint_str).split(",") if inpaint_str is not None else None
        self.inpaint_seq_tensor=inpaint_seq_tensor
        self.inpaint_str_tensor=inpaint_str_tensor
        self.parsed_pdb = parsed_pdb
        self.topo=topo
        if ref_idx is None:
            #using default contig generation, which outputs in rosetta-like format
            self.contigs=contigs
            self.sampled_mask,self.contig_length,self.n_inpaint_chains = self.get_sampled_mask()
            self.receptor_chain = self.chain_order[self.n_inpaint_chains]
            self.receptor, self.receptor_hal, self.receptor_rf, self.inpaint, self.inpaint_hal, self.inpaint_rf= self.expand_sampled_mask()
            self.ref = self.inpaint + self.receptor
            self.hal = self.inpaint_hal + self.receptor_hal
            self.rf = self.inpaint_rf + self.receptor_rf   
        else:
            #specifying precise mappings
            self.ref=ref_idx
            self.hal=hal_idx
            self.rf = rf_idx
        self.mask_1d = [False if i == ('_','_') else True for i in self.ref]
        
        #take care of sequence and structure masking
        if self.inpaint_seq_tensor is None:
            if self.inpaint_seq is not None:
                self.inpaint_seq = self.get_inpaint_seq_str(self.inpaint_seq)
            else:
                self.inpaint_seq = np.array([True if i != ('_','_') else False for i in self.ref])
        else:
            self.inpaint_seq = self.inpaint_seq_tensor

        if self.inpaint_str_tensor is None:
            if self.inpaint_str is not None:
                self.inpaint_str = self.get_inpaint_seq_str(self.inpaint_str)
            else:
                self.inpaint_str = np.array([True if i != ('_','_') else False for i in self.ref])
        else:
            self.inpaint_str = self.inpaint_str_tensor        
        #get 0-indexed input/output (for trb file)
        self.ref_idx0,self.hal_idx0, self.ref_idx0_inpaint, self.hal_idx0_inpaint, self.ref_idx0_receptor, self.hal_idx0_receptor=self.get_idx0()
    
    def get_sampled_mask(self):
        '''
        Function to get a sampled mask from a contig.
        '''
        length_compatible=False
        count = 0
        while length_compatible is False:
            inpaint_chains=0
            contig_list = self.contigs
            sampled_mask = []
            sampled_mask_length = 0
            #allow receptor chain to be last in contig string
            if all([i[0].isalpha() for i in contig_list[-1].split(",")]):
                contig_list[-1] = f'{contig_list[-1]},0'
            for con in contig_list:
                if ((all([i[0].isalpha() for i in con.split(",")[:-1]]) and con.split(",")[-1] == '0')) or self.topo is True:    
                    #receptor chain
                    sampled_mask.append(con)
                else:
                    inpaint_chains += 1
                    #chain to be inpainted. These are the only chains that count towards the length of the contig
                    subcons = con.split(",")
                    subcon_out = []
                    for subcon in subcons:
                        if subcon[0].isalpha():
                            subcon_out.append(subcon)
                            if '-' in subcon:
                                sampled_mask_length += (int(subcon.split("-")[1])-int(subcon.split("-")[0][1:])+1)
                            else:
                                sampled_mask_length += 1

                        else:
                            if '-' in subcon:
                                length_inpaint=random.randint(int(subcon.split("-")[0]),int(subcon.split("-")[1]))
                                subcon_out.append(f'{length_inpaint}-{length_inpaint}')
                                sampled_mask_length += length_inpaint
                            elif subcon == '0':
                                subcon_out.append('0')
                            else:
                                length_inpaint=int(subcon)
                                subcon_out.append(f'{length_inpaint}-{length_inpaint}')
                                sampled_mask_length += int(subcon)
                    sampled_mask.append(','.join(subcon_out))
            #check length is compatible 
            if self.length is not None:
                if sampled_mask_length >= self.length[0] and sampled_mask_length < self.length[1]:
                    length_compatible = True
            else:
                length_compatible = True
            count+=1
            if count == 100000: #contig string incompatible with this length
                sys.exit("Contig string incompatible with --length range")
        return sampled_mask, sampled_mask_length, inpaint_chains

    def expand_sampled_mask(self):
        chain_order='ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        receptor = []
        inpaint = []
        receptor_hal = []
        inpaint_hal = []
        receptor_idx = 1
        inpaint_idx = 1
        inpaint_chain_idx=-1
        receptor_chain_break=[]
        inpaint_chain_break = []
        for con in self.sampled_mask:
            if (all([i[0].isalpha() for i in con.split(",")[:-1]]) and con.split(",")[-1] == '0') or self.topo is True:
                #receptor chain
                subcons = con.split(",")[:-1]
                assert all([i[0] == subcons[0][0] for i in subcons]), "If specifying fragmented receptor in a single block of the contig string, they MUST derive from the same chain"
                assert all(int(subcons[i].split("-")[0][1:]) < int(subcons[i+1].split("-")[0][1:]) for i in range(len(subcons)-1)), "If specifying multiple fragments from the same chain, pdb indices must be in ascending order!"
                for idx, subcon in enumerate(subcons):
                    ref_to_add = [(subcon[0], i) for i in np.arange(int(subcon.split("-")[0][1:]),int(subcon.split("-")[1])+1)]
                    receptor.extend(ref_to_add)
                    receptor_hal.extend([(self.receptor_chain,i) for i in np.arange(receptor_idx, receptor_idx+len(ref_to_add))])
                    receptor_idx += len(ref_to_add)
                    if idx != len(subcons)-1:
                        idx_jump = int(subcons[idx+1].split("-")[0][1:]) - int(subcon.split("-")[1]) -1 
                        receptor_chain_break.append((receptor_idx-1,idx_jump)) #actual chain break in pdb chain
                    else:
                        receptor_chain_break.append((receptor_idx-1,200)) #200 aa chain break 
            else:
                inpaint_chain_idx += 1
                for subcon in con.split(","):
                    if subcon[0].isalpha():
                        ref_to_add=[(subcon[0], i) for i in np.arange(int(subcon.split("-")[0][1:]),int(subcon.split("-")[1])+1)]
                        inpaint.extend(ref_to_add)
                        inpaint_hal.extend([(chain_order[inpaint_chain_idx], i) for i in np.arange(inpaint_idx,inpaint_idx+len(ref_to_add))])
                        inpaint_idx += len(ref_to_add)
                    
                    else:
                        inpaint.extend([('_','_')] * int(subcon.split("-")[0]))
                        inpaint_hal.extend([(chain_order[inpaint_chain_idx], i) for i in np.arange(inpaint_idx,inpaint_idx+int(subcon.split("-")[0]))])
                        inpaint_idx += int(subcon.split("-")[0])
                inpaint_chain_break.append((inpaint_idx-1,200))
    
        if self.topo is True or inpaint_hal == []:
            receptor_hal = [(i[0], i[1]) for i in receptor_hal]
        else:        
            receptor_hal = [(i[0], i[1] + inpaint_hal[-1][1]) for i in receptor_hal] #rosetta-like numbering
        #get rf indexes, with chain breaks
        inpaint_rf = np.arange(0,len(inpaint))
        receptor_rf = np.arange(len(inpaint)+200,len(inpaint)+len(receptor)+200)
        for ch_break in inpaint_chain_break[:-1]:
            receptor_rf[:] += 200
            inpaint_rf[ch_break[0]:] += ch_break[1]
        for ch_break in receptor_chain_break[:-1]:
            receptor_rf[ch_break[0]:] += ch_break[1]
    
        return receptor, receptor_hal, receptor_rf.tolist(), inpaint, inpaint_hal, inpaint_rf.tolist()

    def get_inpaint_seq_str(self, inpaint_s):
        '''
        function to generate inpaint_str or inpaint_seq masks specific to this contig
        '''
        s_mask = np.copy(self.mask_1d)
        inpaint_s_list = []
        for i in inpaint_s:
            if '-' in i:
                inpaint_s_list.extend([(i[0],p) for p in range(int(i.split("-")[0][1:]), int(i.split("-")[1])+1)])
            else:
                inpaint_s_list.append((i[0],int(i[1:])))
        for res in inpaint_s_list:
            if res in self.ref:
                s_mask[self.ref.index(res)] = False #mask this residue
    
        return np.array(s_mask) 

    def get_idx0(self):
        ref_idx0=[]
        hal_idx0=[]
        ref_idx0_inpaint=[]
        hal_idx0_inpaint=[]
        ref_idx0_receptor=[]
        hal_idx0_receptor=[]
        for idx, val in enumerate(self.ref):
            if val != ('_','_'):
                assert val in self.parsed_pdb['pdb_idx'],f"{val} is not in pdb file!"
                hal_idx0.append(idx)
                ref_idx0.append(self.parsed_pdb['pdb_idx'].index(val))
        for idx, val in enumerate(self.inpaint):
            if val != ('_','_'):
                hal_idx0_inpaint.append(idx)
                ref_idx0_inpaint.append(self.parsed_pdb['pdb_idx'].index(val))
        for idx, val in enumerate(self.receptor):
            if val != ('_','_'):
                hal_idx0_receptor.append(idx)
                ref_idx0_receptor.append(self.parsed_pdb['pdb_idx'].index(val))


        return ref_idx0, hal_idx0, ref_idx0_inpaint, hal_idx0_inpaint, ref_idx0_receptor, hal_idx0_receptor

def get_mappings(rm):
    mappings = {}
    mappings['con_ref_pdb_idx'] = [i for i in rm.inpaint if i != ('_','_')]
    mappings['con_hal_pdb_idx'] = [rm.inpaint_hal[i] for i in range(len(rm.inpaint_hal)) if rm.inpaint[i] != ("_","_")]
    mappings['con_ref_idx0'] = rm.ref_idx0_inpaint
    mappings['con_hal_idx0'] = rm.hal_idx0_inpaint
    if rm.inpaint != rm.ref:
        mappings['complex_con_ref_pdb_idx'] = [i for i in rm.ref if i != ("_","_")]
        mappings['complex_con_hal_pdb_idx'] = [rm.hal[i] for i in range(len(rm.hal)) if rm.ref[i] != ("_","_")]
        mappings['receptor_con_ref_pdb_idx'] = [i for i in rm.receptor if i != ("_","_")]
        mappings['receptor_con_hal_pdb_idx'] = [rm.receptor_hal[i] for i in range(len(rm.receptor_hal)) if rm.receptor[i] != ("_","_")]
        mappings['complex_con_ref_idx0'] = rm.ref_idx0
        mappings['complex_con_hal_idx0'] = rm.hal_idx0
        mappings['receptor_con_ref_idx0'] = rm.ref_idx0_receptor
        mappings['receptor_con_hal_idx0'] = rm.hal_idx0_receptor
    mappings['inpaint_str'] = rm.inpaint_str
    mappings['inpaint_seq'] = rm.inpaint_seq
    mappings['sampled_mask'] = rm.sampled_mask
    mappings['mask_1d'] = rm.mask_1d
    return mappings

def lddt_unbin(pred_lddt):
    nbin = pred_lddt.shape[1]
    bin_step = 1.0 / nbin
    lddt_bins = torch.linspace(bin_step, 1.0, nbin, dtype=pred_lddt.dtype, device=pred_lddt.device)

    pred_lddt = nn.Softmax(dim=1)(pred_lddt)
    return torch.sum(lddt_bins[None,:,None]*pred_lddt, dim=1)

