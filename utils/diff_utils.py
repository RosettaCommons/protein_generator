import torch
from icecream import ic
import random
import numpy as np
from kinematics import get_init_xyz
import torch.nn as nn 
from util_module import ComputeAllAtomCoords
from util import *
from inpainting_util import MSAFeaturize_fixbb, TemplFeaturizeFixbb, lddt_unbin
from kinematics import xyz_to_t2d


def mask_inputs(seq, msa_masked, msa_full, xyz_t, t1d, input_seq_mask=None, 
        input_str_mask=None, input_t1dconf_mask=None, diffuser=None, t=None, 
        MODEL_PARAM=None, hotspots=None, dssp=None, v2_mode=False):


    """
    JG - adapted slightly for the inference case

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
    assert diffuser != None, 'please choose a diffuser'

    ###########
    seq = seq[0,:1]
    msa_masked = msa_masked[0,:1]
    msa_full = msa_full[0,:1]
    t1d = t1d[0]
    xyz_t = xyz_t[0]

    seq_mask = input_seq_mask[0]



    ######################
    ###sequence diffusion###
    ######################
    """
    #muate some percentage of sequence to have model be able to mutate residues later in denoising trajectory
    if True:
        masked_values=input_seq_mask[0].nonzero()[:,0]
        print(masked_values)
        mut_p=math.floor(masked_values.shape[0]*.05)
        print(mut_p)
        mutate_indices = torch.randperm(len(masked_values))[:mut_p]
        print(mutate_indices)
        for i in range(len(mutate_indices)):
            seq[0,masked_values[mutate_indices[i]]]  = torch.randint(0, 21, (1,))
    """
    str_mask     = input_str_mask[0]
    
    x_0          = torch.nn.functional.one_hot(seq[0,...],num_classes=22).float()*2-1
    
    #ic(seq_mask)

    seq_diffused = diffuser.q_sample(x_0,torch.tensor([t-1]),mask=seq_mask)
    #seq_diffused = torch.clamp(seq_diffused, min=-1, max=1)

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
    msa_diffused = diffuser.q_sample(x_0_msa,torch.tensor([t-1]),mask=msa_seq_mask)
    #msa_diffused = torch.clamp(msa_diffused, min=-1, max=1)
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
    #msa_full[:,0,:,:22] = seq_diffused
    #make msa_full same size as msa_masked
    msa_full = msa_full[:,:msa_masked.shape[1],:,:]
    msa_full[:,0,:,:22] = seq_diffused
    msa_full[:,1:,:,:22] = msa_diffused

    ### t1d ###
    ########### 
    # NOTE: adjusting t1d last dim (confidence) from sequence mask
    t1d = torch.cat((t1d, torch.zeros((t1d.shape[0],t1d.shape[1],2)).float()), -1).to(seq.device)
    t1d[:,:,:21] = seq_diffused[...,:21]

    #t1d[:,:,21] *= input_t1dconf_mask
    #set diffused conf to 0 and everything else to 1
    t1d[:,~seq_mask,21] = 0.0
    t1d[:,seq_mask,21] = 1.0
    
    t1d[:1,:,22] = 1-t/diffuser.num_timesteps

    t1d[:,~str_mask,23] = 0.0
    t1d[:,str_mask,23] = 1.0
    
    # EXPAND t1d to match model params
    if MODEL_PARAM['d_t1d'] == 29:
        ## added t1d features ##
        # 24 -- dssp helix
        # 25 -- dssp sheet
        # 26 -- dssp loop
        # 27 -- dssp mask
        # 28 -- hotspot resi on target
        t1d = torch.cat((t1d,torch.zeros(t1d.shape[0],t1d.shape[1],5)),dim=-1)
        t1d[:,:,24:28] = dssp
        t1d[:,:,28] = hotspots
        t1d[:,str_mask,24:27] = 0.0
        t1d[:,str_mask,27] = 1.0

    xyz_t = get_init_xyz(xyz_t[None])
    xyz_t = xyz_t[0]

    xyz_t[:,~seq_mask,3:,:] = float('nan')

    # Structure masking
    xyz_t[:,~str_mask,:,:] = float('nan')

    if not v2_mode:
        xyz_t = get_init_xyz(xyz_t[None])
        xyz_t = xyz_t[0]
        assert torch.sum(torch.isnan(xyz_t[:,:,:3,:]))==0

    return seq, msa_masked, msa_full, xyz_t, t1d, seq_diffused


conversion = 'ARNDCQEGHILKMFPSTWYVX-'


def take_step(model, msa, msa_extra, seq, t1d, t2d, idx_pdb, N_cycle, xyz_prev, alpha, xyz_t, 
        alpha_t, seq_diffused, msa_prev, pair_prev, state_prev):
    """ 
    Single step in the diffusion process
    """
    compute_allatom_coords=ComputeAllAtomCoords().to(seq.device) 
    #ic(msa.shape)
    B, _, N, L, _ = msa.shape
    with torch.no_grad():
        with torch.cuda.amp.autocast(True):
            for i_cycle in range(N_cycle-1):
                msa_prev, pair_prev, xyz_prev, state_prev, alpha = model(msa[:,0],
                                                                   msa_extra[:,0],
                                                                   seq[:,0], xyz_prev,
                                                                   idx_pdb,
                                                                   seq1hot=seq_diffused,
                                                                   t1d=t1d, t2d=t2d,
                                                                   xyz_t=xyz_t, alpha_t=alpha_t,
                                                                   msa_prev=msa_prev,
                                                                   pair_prev=pair_prev,
                                                                   state_prev=state_prev,
                                                                   return_raw=True) 
                
            
            logit_s, logit_aa_s, logits_exp, xyz_prev, pred_lddt, msa_prev, pair_prev, state_prev, alpha = model(msa[:,0], 
                                                            msa_extra[:,0],
                                                            seq[:,0], xyz_prev,
                                                            idx_pdb,
                                                            seq1hot=seq_diffused,
                                                            t1d=t1d, t2d=t2d, xyz_t=xyz_t, alpha_t=alpha_t,
                                                            msa_prev=msa_prev,
                                                            pair_prev=pair_prev,
                                                            state_prev=state_prev,
                                                            return_infer=True)
        logit_aa_s_msa = torch.clone(logit_aa_s)
        logit_aa_s = logit_aa_s.reshape(B,-1,N,L)[:,:,0,:]
        logit_aa_s = logit_aa_s.reshape(B,-1,L)
        seq_out = torch.argmax(logit_aa_s, dim=-2)

        pred_lddt_unbinned = lddt_unbin(pred_lddt)
        _, xyz_prev = compute_allatom_coords(seq_out, xyz_prev, alpha)
    
    if N>1:
        return seq_out, xyz_prev, pred_lddt_unbinned, logit_s, logit_aa_s, logit_aa_s_msa, alpha, msa_prev, pair_prev, state_prev  
    else:
        return seq_out, xyz_prev, pred_lddt_unbinned, logit_s, logit_aa_s, alpha, msa_prev, pair_prev, state_prev
            
            
def take_step_nostate(model, msa, msa_extra, seq, t1d, t2d, idx_pdb, N_cycle, xyz_prev, alpha, xyz_t,
        alpha_t, seq_diffused, msa_prev, pair_prev, state_prev):
    """ 
    Single step in the diffusion process, with no conditioning on state
    """
    compute_allatom_coords=ComputeAllAtomCoords().to(seq.device)
    msa_prev = None
    pair_prev = None
    state_prev = None
    
    B, _, N, L, _ = msa.shape
    with torch.no_grad():
        with torch.cuda.amp.autocast(True):
            for i_cycle in range(N_cycle-1):
                msa_prev, pair_prev, xyz_prev, state_prev, alpha = model(msa[:,0],
                                                                   msa_extra[:,0],
                                                                   seq[:,0], xyz_prev,
                                                                   idx_pdb,
                                                                   seq1hot=seq_diffused,
                                                                   t1d=t1d, t2d=t2d,
                                                                   xyz_t=xyz_t, alpha_t=alpha_t,
                                                                   msa_prev=msa_prev,
                                                                   pair_prev=pair_prev,
                                                                   state_prev=state_prev,
                                                                   return_raw=True)


            logit_s, logit_aa_s, logits_exp, xyz_prev, pred_lddt, msa_prev, pair_prev, state_prev, alpha = model(msa[:,0],
                                                            msa_extra[:,0],
                                                            seq[:,0], xyz_prev,
                                                            idx_pdb,
                                                            seq1hot=seq_diffused,
                                                            t1d=t1d, t2d=t2d, xyz_t=xyz_t, alpha_t=alpha_t,
                                                            msa_prev=msa_prev,
                                                            pair_prev=pair_prev,
                                                            state_prev=state_prev,
                                                            return_infer=True)

        logit_aa_s_msa = torch.clone(logit_aa_s)
        logit_aa_s = logit_aa_s.reshape(B,-1,N,L)[:,:,0,:]
        logit_aa_s = logit_aa_s.reshape(B,-1,L)
        seq_out = torch.argmax(logit_aa_s, dim=-2)

        pred_lddt_unbinned = lddt_unbin(pred_lddt)
        _, xyz_prev = compute_allatom_coords(seq_out, xyz_prev, alpha)

    if N>1:
        return seq_out, xyz_prev, pred_lddt_unbinned, logit_s, logit_aa_s, logit_aa_s_msa, alpha, msa_prev, pair_prev, state_prev
    else:
        return seq_out, xyz_prev, pred_lddt_unbinned, logit_s, logit_aa_s, alpha, msa_prev, pair_prev, state_prev


def get_alphas(t1d, xyz_t, B, L, ti_dev, ti_flip, ang_ref):
    # get torsion angles from templates
    seq_tmp = t1d[...,:21].argmax(dim=-1).reshape(-1,L)
    alpha, _, alpha_mask, _ = get_torsions(xyz_t.reshape(-1,L,27,3), seq_tmp, ti_dev, ti_flip, ang_ref)
    alpha_mask = torch.logical_and(alpha_mask, ~torch.isnan(alpha[...,0]))
    alpha[torch.isnan(alpha)] = 0.0
    alpha = alpha.reshape(B,-1,L,10,2)
    alpha_mask = alpha_mask.reshape(B,-1,L,10,1)
    alpha_t = torch.cat((alpha, alpha_mask), dim=-1).reshape(B, -1, L, 30)
    return alpha, alpha_t
