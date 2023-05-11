import sys, os
import torch
from icecream import ic
import random
import numpy as np
from kinematics import get_init_xyz
sys.path.append('../')
from utils.calc_dssp import annotate_sse

ic.configureOutput(includeContext=True)

def mask_inputs(seq, 
        msa_masked, 
        msa_full, 
        xyz_t, 
        t1d, 
        mask_msa, 
        input_seq_mask=None, 
        input_str_mask=None, 
        input_floating_mask=None, 
        input_t1dconf_mask=None, 
        loss_seq_mask=None, 
        loss_str_mask=None, 
        loss_str_mask_2d=None, 
        dssp=False,
        hotspots=False,
        diffuser=None, 
        t=None, 
        freeze_seq_emb=False, 
        mutate_seq=False, 
        no_clamp_seq=False, 
        norm_input=False,
        contacts=None,
        frac_provide_dssp=0.5,
        dssp_mask_percentage=[0,100],
        frac_provide_contacts=0.5,
        struc_cond=False):
    """
    Parameters:
        seq (torch.tensor, required): (I,L) integer sequence 

        msa_masked (torch.tensor, required): (I,N_short,L,48)

        msa_full  (torch,.tensor, required): (I,N_long,L,25)
        
        xyz_t (torch,tensor): (T,L,27,3) template crds BEFORE they go into get_init_xyz 
        
        t1d (torch.tensor, required): (I,L,22) this is the t1d before tacking on the chi angles 
        
        str_mask_1D (torch.tensor, required): Shape (L) rank 1 tensor where structure is masked at False positions 

        seq_mask_1D (torch.tensor, required): Shape (L) rank 1 tensor where seq is masked at False positions
        t1d_24: is there an extra dimension to input structure confidence?

        diffuser: diffuser class
        
        t: time step

    NOTE: in the MSA, the order is 20aa, 1x unknown, 1x mask token. We set the masked region to 22 (masked).
        For the t1d, this has 20aa, 1x unkown, and 1x template conf. Here, we set the masked region to 21 (unknown).
        This, we think, makes sense, as the template in normal RF training does not perfectly correspond to the MSA.
    """

    
    
    #ic(input_seq_mask.shape)
    #ic(seq.shape)
    #ic(msa_masked.shape)
    #ic(msa_full.shape)
    #ic(t1d.shape)
    #ic(xyz_t.shape)
    #ic(input_str_mask.shape)
    #ic(mask_msa.shape)

    ###########
    seq_mask = input_seq_mask


    ######################
    ###sequence diffusion###
    ######################
    
    str_mask     = input_str_mask
    
    x_0          = torch.nn.functional.one_hot(seq[0,...],num_classes=22).float()*2-1
    seq_diffused = diffuser.q_sample(x_0,t,mask=seq_mask)
    
    seq_tmp=torch.argmax(seq_diffused,axis=-1).to(device=seq.device)
    seq=seq_tmp.repeat(seq.shape[0], 1)

    ###################
    ###msa diffusion###
    ###################

    ### msa_masked ###
    #ic(msa_masked.shape)
    B,N,L,_=msa_masked.shape
    msa_masked[:,0,:,:22] = seq_diffused

    x_0_msa = msa_masked[0,1:,:,:22].float()*2-1
    msa_seq_mask = seq_mask.unsqueeze(0).repeat(N-1, 1)
    msa_diffused = diffuser.q_sample(x_0_msa,torch.tensor([t]),mask=msa_seq_mask)
    
    msa_masked[:,1:,:,:22] = torch.clone(msa_diffused)

    # index 44/45 is insertion/deletion
    # index 43 is the masked token NOTE check this
    # index 42 is the unknown token 
    msa_masked[:,0,:,22:44] = seq_diffused
    msa_masked[:,1:,:,22:44] = msa_diffused

    # insertion/deletion stuff 
    msa_masked[:,0,~seq_mask,44:46] = 0

    ### msa_full ### 
    ################
    #make msa_full same size as msa_masked
    #ic(msa_full.shape)
    msa_full = msa_full[:,:msa_masked.shape[1],:,:]
    msa_full[:,0,:,:22] = seq_diffused
    msa_full[:,1:,:,:22] = msa_diffused

    ### t1d ###
    ########### 
    # NOTE: adjusting t1d last dim (confidence) from sequence mask
    t1d = torch.cat((t1d, torch.zeros((t1d.shape[0],t1d.shape[1],1)).float()), -1).to(seq.device)
    t1d[:,:,:21] = seq_diffused[...,:21]

    #t1d[:,:,21] *= input_t1dconf_mask
    #set diffused conf to 0 and everything else to 1
    t1d[:,~seq_mask,21] = 0.0
    t1d[:,seq_mask,21] = 1.0

    t1d[:1,:,22] = 1-t/diffuser.num_timesteps
    
    #to do add structure confidence metric; need to expand dimensions of chkpt b4
    #if t1d_24: JG - changed to be default
    t1d = torch.cat((t1d, torch.zeros((t1d.shape[0],t1d.shape[1],1)).float()), -1).to(seq.device)
    t1d[:,~str_mask,23] = 0.0
    t1d[:,str_mask,23] = 1.0

    if dssp:
        print(f'adding dssp {frac_provide_dssp} of time')
        t1d = torch.cat((t1d, torch.zeros((t1d.shape[0],t1d.shape[1],4)).float()), -1).to(seq.device)
        #dssp info
        #mask some percentage of dssp info in range dssp_mask_percentage[0],dssp_mask_percentage[1]
        percentage_mask=random.randint(dssp_mask_percentage[0], dssp_mask_percentage[1])
        dssp=annotate_sse(np.array(xyz_t[0,:,1,:].squeeze()), percentage_mask=percentage_mask)
        #dssp_unmasked = annotate_sse(np.array(xyz_t[0,:,1,:].squeeze()), percentage_mask=0)
        if np.random.rand()>frac_provide_dssp:
            print('masking dssp')
            dssp[...]=0 #replace with mask token
            dssp[:,-1]=1
        t1d[...,24:]=dssp
    
    if hotspots:
        print(f"adding hotspots {frac_provide_contacts} of time")
        t1d = torch.cat((t1d, torch.zeros((t1d.shape[0],t1d.shape[1],1)).float()), -1).to(seq.device)
        #mask all contacts some fraction of the time
        if np.random.rand()>frac_provide_contacts:
            print('masking contacts')
            contacts = torch.zeros(L) 
        t1d[...,-1] = contacts

    ### xyz_t ###
    #############
    xyz_t = get_init_xyz(xyz_t[None])
    xyz_t = xyz_t[0]
    #Sequence masking
    xyz_t[:,:,3:,:] = float('nan')
    # Structure masking
    if struc_cond:
        print("non-autoregressive structure conditioning")
        r = diffuser.alphas_cumprod[t]
        xyz_mask = (torch.rand(xyz_t.shape[1]) > r).to(torch.bool).to(seq.device)
        xyz_mask = torch.logical_and(xyz_mask,~str_mask)
        xyz_t[:,xyz_mask,:,:] = float('nan')
    else:
        xyz_t[:,~str_mask,:,:] = float('nan')
    
    ### mask_msa ###
    ################
    # NOTE: this is for loss scoring
    mask_msa[:,:,~loss_seq_mask] = False
    
    out=dict(
            seq= seq,
            msa_masked= msa_masked,
            msa_full= msa_full,
            xyz_t= xyz_t,
            t1d= t1d,
            mask_msa= mask_msa,
            seq_diffused= seq_diffused
            )
    
    return out
