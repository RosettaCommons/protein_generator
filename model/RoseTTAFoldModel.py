import torch
import torch.nn as nn
from Embeddings import MSA_emb, Extra_emb, Templ_emb, Recycling
from Track_module import IterativeSimulator
from AuxiliaryPredictor import DistanceNetwork, MaskedTokenNetwork, ExpResolvedNetwork, LDDTNetwork
from util import INIT_CRDS
from opt_einsum import contract as einsum
from icecream import ic

class RoseTTAFoldModule(nn.Module):
    def __init__(self, n_extra_block=4, n_main_block=8, n_ref_block=4,\
                 d_msa=256, d_msa_full=64, d_pair=128, d_templ=64,
                 n_head_msa=8, n_head_pair=4, n_head_templ=4,
                 d_hidden=32, d_hidden_templ=64,
                 p_drop=0.15, d_t1d=24, d_t2d=44,
                 SE3_param_full={'l0_in_features':32, 'l0_out_features':16, 'num_edge_features':32},
                 SE3_param_topk={'l0_in_features':32, 'l0_out_features':16, 'num_edge_features':32},
                 ):
        super(RoseTTAFoldModule, self).__init__()
        #
        # Input Embeddings
        d_state = SE3_param_topk['l0_out_features']
        self.latent_emb = MSA_emb(d_msa=d_msa, d_pair=d_pair, d_state=d_state, p_drop=p_drop)
        self.full_emb = Extra_emb(d_msa=d_msa_full, d_init=25, p_drop=p_drop)
        self.templ_emb = Templ_emb(d_pair=d_pair, d_templ=d_templ, d_state=d_state,
                                   n_head=n_head_templ,
                                   d_hidden=d_hidden_templ, p_drop=0.25, d_t1d=d_t1d, d_t2d=d_t2d)
        # Update inputs with outputs from previous round
        self.recycle = Recycling(d_msa=d_msa, d_pair=d_pair, d_state=d_state)
        #
        self.simulator = IterativeSimulator(n_extra_block=n_extra_block,
                                            n_main_block=n_main_block,
                                            n_ref_block=n_ref_block,
                                            d_msa=d_msa, d_msa_full=d_msa_full,
                                            d_pair=d_pair, d_hidden=d_hidden,
                                            n_head_msa=n_head_msa,
                                            n_head_pair=n_head_pair,
                                            SE3_param_full=SE3_param_full,
                                            SE3_param_topk=SE3_param_topk,
                                            p_drop=p_drop)
        ##
        self.c6d_pred = DistanceNetwork(d_pair, p_drop=p_drop)
        self.aa_pred = MaskedTokenNetwork(d_msa, p_drop=p_drop)
        self.lddt_pred = LDDTNetwork(d_state)
       
        self.exp_pred = ExpResolvedNetwork(d_msa, d_state)

    def forward(self, msa_latent, msa_full, seq, xyz, idx,
                seq1hot=None, t1d=None, t2d=None, xyz_t=None, alpha_t=None,
                msa_prev=None, pair_prev=None, state_prev=None,
                return_raw=False, return_full=False,
                use_checkpoint=False, return_infer=False):
        B, N, L = msa_latent.shape[:3]
        # Get embeddings
        #ic(seq.shape)
        #ic(msa_latent.shape)
        #ic(seq1hot.shape)
        #ic(idx.shape)
        #ic(xyz.shape)
        #ic(seq1hot.shape)
        #ic(t1d.shape)
        #ic(t2d.shape)

        idx = idx.long()
        msa_latent, pair, state = self.latent_emb(msa_latent, seq, idx, seq1hot=seq1hot)

        print(f'pair.grad_fn: {pair.grad_fn}')
        print(f'pair.requires_grad: {pair.requires_grad}')
        msa_full = self.full_emb(msa_full, seq, idx, seq1hot=seq1hot)
        # with torch.no_grad():
        # Do recycling
        if msa_prev == None:
            msa_prev = torch.zeros_like(msa_latent[:,0])
        if pair_prev == None:
            pair_prev = torch.zeros_like(pair)
        if state_prev == None:
            state_prev = torch.zeros_like(state)

        #ic(seq.shape)
        #ic(msa_prev.shape)
        #ic(pair_prev.shape)
        #ic(xyz.shape)
        #ic(state_prev.shape)


        msa_recycle, pair_recycle, state_recycle = self.recycle(seq, msa_prev, pair_prev, xyz, state_prev)
        msa_latent[:,0] = msa_latent[:,0] + msa_recycle.reshape(B,L,-1)
        pair = pair + pair_recycle
        state = state + state_recycle
        #
        #ic(t1d.dtype)
        #ic(t2d.dtype)
        #ic(alpha_t.dtype)
        #ic(xyz_t.dtype)
        #ic(pair.dtype)
        #ic(state.dtype)


        #import pdb; pdb.set_trace()
        
        # add template embedding
        pair, state = self.templ_emb(t1d, t2d, alpha_t, xyz_t, pair, state, use_checkpoint=use_checkpoint)
        
        #ic(seq.dtype)
        #ic(msa_latent.dtype)
        #ic(msa_full.dtype)
        #ic(pair.dtype)
        #ic(xyz.dtype)
        #ic(state.dtype)
        #ic(idx.dtype)

        # Predict coordinates from given inputs
        msa, pair, R, T, alpha_s, state = self.simulator(seq, msa_latent, msa_full.type(torch.float32), pair, xyz[:,:,:3],
                                                         state, idx, use_checkpoint=use_checkpoint)
        
        if return_raw:
            # get last structure
            xyz = einsum('bnij,bnaj->bnai', R[-1], xyz[:,:,:3]-xyz[:,:,1].unsqueeze(-2)) + T[-1].unsqueeze(-2)
            return msa[:,0], pair, xyz, state, alpha_s[-1]

        # predict masked amino acids
        logits_aa = self.aa_pred(msa)
        #
        # predict distogram & orientograms
        logits = self.c6d_pred(pair)
        
        # Predict LDDT
        with torch.no_grad():
            lddt = self.lddt_pred(state)

        # predict experimentally resolved or not
        logits_exp = self.exp_pred(msa[:,0], state)
        
        if return_infer:
            #get last structure
            xyz = einsum('bnij,bnaj->bnai', R[-1], xyz[:,:,:3]-xyz[:,:,1].unsqueeze(-2)) + T[-1].unsqueeze(-2)
            return logits, logits_aa, logits_exp, xyz, lddt, msa[:,0], pair, state, alpha_s[-1]


        # get all intermediate bb structures
        xyz = einsum('rbnij,bnaj->rbnai', R, xyz[:,:,:3]-xyz[:,:,1].unsqueeze(-2)) + T.unsqueeze(-2)

        return logits, logits_aa, logits_exp, xyz, alpha_s, lddt
