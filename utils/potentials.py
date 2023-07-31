import os, sys
import shutil
import glob
import torch
import numpy as np
import copy
from itertools import groupby
from operator import itemgetter
import json
import re
import random
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import random
import Bio
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

conversion = 'ARNDCQEGHILKMFPSTWYVX-'


### IF ADDING NEW POTENTIAL MAKE SURE TO ADD TO BOTTOM DICTIONARY ###


# TEMPLATE CLASS
class Potential:
    
    def get_gradients(seq):
        '''
            EVERY POTENTIAL CLASS MUST RETURN GRADIENTS
        '''
        
        sys.exit('ERROR POTENTIAL HAS NOT BEEN IMPLEMENTED')
    
    
class AACompositionalBias(Potential):
    """
    T = number of timesteps to set up diffuser with
    
    schedule = type of noise schedule to use linear, cosine, gaussian
    
    noise = type of ditribution to sample from; DEFAULT - normal_gaussian
    
    """

    def __init__(self, args, features, potential_scale, DEVICE): 
        
        self.L = features['L']
        self.DEVICE = DEVICE
        self.frac_seq_to_weight = args['frac_seq_to_weight']
        self.add_weight_every_n = args['add_weight_every_n']
        self.aa_weights_json = args['aa_weights_json']
        self.one_weight_per_position = args['one_weight_per_position']
        self.aa_weight = args['aa_weight']
        self.aa_spec = args['aa_spec']
        self.aa_composition = args['aa_composition']
        self.potential_scale = potential_scale
        
        self.aa_weights_to_add = [0 for l in range(21)]
        self.aa_max_potential = None
        
        
        if self.aa_weights_json != None:
            with open(self.aa_weights_json, 'r') as f:
                aa_weights = json.load(f)
        else:
            aa_weights = {}

        for k,v in aa_weights.items():
            aa_weights_to_add[conversion.index(k)] = v
        
        aa_weights_to_add = [0 for l in range(21)]
        
        self.aa_weights_to_add = torch.tensor(aa_weights_to_add)[None].repeat(self.L,1).to(self.DEVICE, non_blocking=True)

        # BLOCK TO FIND OUT HOW YOU ARE LOOKING TO PROVIDE AA COMPOSITIONAL BIAS
        if self.add_weight_every_n > 1 or self.frac_seq_to_weight > 0:
            
            assert (self.add_weight_every_n > 1) ^ (self.frac_seq_to_weight > 0), 'use either --add_weight_every_n or --frac_seq_to_weight but not both'
            weight_mask = torch.zeros_like(self.aa_weights_to_add)
            if add_weight_every_n > 1:
                idxs_to_unmask = torch.arange(0,self.L,self.add_weight_every_n)
            else:
                indexs = np.arange(0,self.L).tolist()
                idxs_to_unmask = random.sample(indexs,int(self.frac_seq_to_weight*self.L))
                idxs_to_unmask.sort()

            weight_mask[idxs_to_unmask,:] = 1
            self.aa_weights_to_add *= weight_mask

            if one_weight_per_position:
                for p in range(self.aa_weights_to_add.shape[0]):
                    where_ones = torch.where(self.aa_weights_to_add[p,:] > 0)[0].tolist()
                    if len(where_ones) > 0:
                        w_sample = random.sample(where_ones,1)[0]
                        self.aa_weights_to_add[p,:w_sample] = 0
                        self.aa_weights_to_add[p,w_sample+1:] = 0

        elif self.aa_spec != None:

            assert self.aa_weight != None, 'please specify --aa_weight'
            # Use specified repeat structure to bias sequence

            repeat_len = len(self.aa_spec)
            weight_split = [float(x) for x in self.aa_weight.split(',')]

            aa_idxs = []
            for k,c in enumerate(self.aa_spec):
                if c != 'X':
                    assert c in conversion, f'the letter you have chosen is not an amino acid: {c}'
                    aa_idxs.append((k,conversion.index(c)))

            if len(self.aa_weight) > 1:
                assert len(aa_idxs) == len(weight_split), f'need to give same number of weights as AAs in weight spec'

            self.aa_weights_to_add = torch.zeros(self.L,21)

            for p,w in zip(aa_idxs,weight_split):
                x,a = p
                self.aa_weights_to_add[x,a] = w

            self.aa_weights_to_add = self.aa_weights_to_add[:repeat_len,:].repeat(self.L//repeat_len+1,1)[:self.L].to(self.DEVICE, non_blocking=True)

        elif self.aa_composition != None:
            
            self.aa_comp = [(x[0],float(x[1:])) for x in self.aa_composition.split(',')]
            self.aa_max_potential = 0 #just a place holder so not None 
            assert sum([f for aa,f in self.aa_comp]) <= 1, f'total sequence fraction specified in aa_composition is > 1'
            
        else:
            sys.exit(f'You are missing an argument to use the aa_bias potential')
    
    def get_gradients(self, seq):
        '''
            seq = L,21 
            
            return gradients to update the sequence with for the next pass
        '''
        
        if self.aa_max_potential != None:
            soft_seq = torch.softmax(seq, dim=1)
            print('ADDING SOFTMAXED SEQUENCE POTENTIAL')

            aa_weights_to_add_list = []
            for aa,f in self.aa_comp:
                aa_weights_to_add_copy = self.aa_weights_to_add.clone()

                soft_seq_tmp = soft_seq.clone().detach().requires_grad_(True)
                aa_idx = conversion.index(aa)

                # get top-k probability of logit to add to
                where_add = torch.topk(soft_seq_tmp[:,aa_idx], int(f*self.L))[1]

                # set up aa_potenital
                aa_potential = torch.zeros(21)
                aa_potential[conversion.index(aa)] = 1.0
                aa_potential = aa_potential.repeat(self.L,1).to(self.DEVICE, non_blocking=True)

                # apply "loss"
                aa_comp_loss = torch.sum(torch.sum((aa_potential - soft_seq_tmp)**2, dim=1)**0.5)

                # get gradients
                aa_comp_loss.backward()
                update_grads = soft_seq_tmp.grad

                for k in range(self.L):
                    if k in where_add:
                        aa_weights_to_add_copy[k,:] = -update_grads[k,:]*self.potential_scale
                    else:
                        aa_weights_to_add_copy[k,:] = update_grads[k,:]*self.potential_scale
                aa_weights_to_add_list.append(aa_weights_to_add_copy)
                
            aa_weights_to_add_array = torch.stack((aa_weights_to_add_list))
            self.aa_weights_to_add = torch.mean(aa_weights_to_add_array.float(), 0)
            
            
        return self.aa_weights_to_add


class HydrophobicBias(Potential):
    """
    Calculate loss with respect to soft_seq of the sequence hydropathy index (Kyte and Doolittle, 1986).
    
    T = number of timesteps to set up diffuser with
    
    schedule = type of noise schedule to use linear, cosine, gaussian
    
    noise = type of ditribution to sample from; DEFAULT - normal_gaussian
    
    """    
    def __init__(self, args, features, potential_scale, DEVICE):
        
        self.target_score = args['hydrophobic_score']
        self.potential_scale = potential_scale
        self.loss_type = args['hydrophobic_loss_type']
        print(f'USING {self.loss_type} LOSS TYPE...')
        
        # -----------------------------------------------------------------------
        # ---------------------GRAVY index data structures-----------------------
        # -----------------------------------------------------------------------
        
        # AA conversion
        self.alpha_1 = list("ARNDCQEGHILKMFPSTWYVX")

        # Dictionary to convert amino acids to their hyropathy index
        self.gravy_dict = {'C': 2.5, 'D': -3.5, 'S': -0.8, 'Q': -3.5, 'K': -3.9,
        'I': 4.5, 'P': -1.6, 'T': -0.7, 'F': 2.8, 'N': -3.5, 
        'G': -0.4, 'H': -3.2, 'L': 3.8, 'R': -4.5, 'W': -0.9, 
        'A': 1.8, 'V':4.2, 'E': -3.5, 'Y': -1.3, 'M': 1.9, 'X': 0, '-': 0}

        self.gravy_list = [self.gravy_dict[a] for a in self.alpha_1]        
        
        # -----------------------------------------------------------------------
        # -----------------------------------------------------------------------

        print(f'GUIDING SEQUENCES TO HAVE TARGET GRAVY SCORE OF: {self.target_score}')
        return None

        
    def get_gradients(self, seq):
        """
        Calculate gradients with respect to GRAVY index of input seq.
        Uses a MSE loss.

        Arguments
        ---------
        seq : tensor
            L X 21 logits after saving seq_out from xt

        Returns
        -------
        gradients : list of tensors
            gradients of soft_seq with respect to loss on partial_charge
        """
        # Get GRAVY matrix based on length of seq
        gravy_matrix  = torch.tensor(self.gravy_list)[None].repeat(seq.shape[0],1).to(DEVICE)

        # Get softmax of seq
        soft_seq = torch.softmax(seq,dim=-1).requires_grad_(requires_grad=True).to(DEVICE)

        # Calculate simple MSE loss on gravy_score
        if self.loss_type == 'simple':
            gravy_score = torch.mean(torch.sum(soft_seq*gravy_matrix,dim=-1), dim=0)
            loss = ((gravy_score - self.target_score)**2)**0.5
            #print(f'LOSS: {loss}')
            # Take backward step
            loss.backward()

            # Get gradients from soft_seq
            self.gradients = soft_seq.grad
            # plt.imshow(self.gradients.cpu().detach().numpy())
            # plt.colorbar()
            # plt.title('gradients')              

        # Calculate MSE loss on gravy_score
        elif self.loss_type == 'complex':
            loss = torch.mean((torch.sum(soft_seq*gravy_matrix, dim = -1) - self.target_score)**2)
            #print(f'LOSS: {loss}')
            # Take backward step
            loss.backward()

            # Get gradients from soft_seq
            self.gradients = soft_seq.grad
            # plt.imshow(self.gradients.cpu().detach().numpy())
            # plt.colorbar()
            # plt.title('gradients')        

        return -self.gradients*self.potential_scale            
            
        
class ChargeBias(Potential):
    """
    Calculate losses and get gradients with respect to soft_seq for the sequence charge at a given pH.
    
    T = number of timesteps to set up diffuser with
    
    schedule = type of noise schedule to use linear, cosine, gaussian
    
    noise = type of ditribution to sample from; DEFAULT - normal_gaussian
    
    """        
    def __init__(self, args, features, potential_scale, DEVICE):

        self.target_charge = args['target_charge']
        self.pH = args['target_pH']
        self.loss_type = args['charge_loss_type']
        self.potential_scale = potential_scale
        self.L = features['L']
        self.DEVICE = DEVICE
        
        # -----------------------------------------------------------------------
        # ------------------------pI data structures-----------------------------
        # -----------------------------------------------------------------------
        
        # pKa lists to account for every residue.
        pos_pKs_list = [[0.0, 12.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.98, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
        neg_pKs_list = [[0.0, 0.0, 0.0, 4.05, 9.0, 0.0, 4.45, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0]]
        cterm_pKs_list = [[0.0, 0.0, 0.0, 4.55, 0.0, 0.0, 4.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
        nterm_pKs_list = [[7.59, 0.0, 0.0, 0.0, 0.0, 0.0, 7.7, 0.0, 0.0, 0.0, 0.0, 0.0, 7.0, 0.0, 8.36, 6.93, 6.82, 0.0, 0.0, 7.44, 0.0]]

        # Convert pKa lists to tensors 
        self.cterm_pKs = torch.tensor(cterm_pKs_list)
        self.nterm_pKs = torch.tensor(nterm_pKs_list)
        self.pos_pKs = torch.tensor(pos_pKs_list)
        self.neg_pKs = torch.tensor(neg_pKs_list)
        
        # Repeat charged pKs L - 2 times to populate in all non-terminal residue indices
        pos_pKs_repeat = self.pos_pKs.repeat(self.L - 2, 1)
        neg_pKs_repeat = self.neg_pKs.repeat(self.L - 2, 1)

        # Concatenate all pKs tensors with N-term and C-term pKas to get full L X 21 charge matrix
        self.pos_pKs_matrix = torch.cat((torch.zeros_like(self.nterm_pKs), pos_pKs_repeat, self.nterm_pKs)).to(DEVICE)
        self.neg_pKs_matrix = torch.cat((self.cterm_pKs, neg_pKs_repeat, torch.zeros_like(self.cterm_pKs))).to(DEVICE)    

        # Get indices of positive, neutral, and negative residues
        self.cterm_charged_idx = torch.nonzero(self.cterm_pKs)
        self.cterm_neutral_idx = torch.nonzero(self.cterm_pKs == 0)
        self.nterm_charged_idx = torch.nonzero(self.nterm_pKs)
        self.nterm_neutral_idx = torch.nonzero(self.nterm_pKs == 0)
        self.pos_pKs_idx = torch.tensor([[1, 8, 11]])
        self.neg_pKs_idx = torch.tensor([[3, 4, 6, 18]])
        self.neutral_pKs_idx = torch.tensor([[0, 2, 5, 7, 9, 10, 12, 13, 14, 15, 16, 17, 19, 20]])
        
        # -----------------------------------------------------------------------
        # -----------------------------------------------------------------------        
        
        print(f'OPTIMIZING SEQUENCE TO HAVE CHARGE = {self.target_charge}\nAT pH = {self.pH}' )

    def sum_tensor_indices(self, indices, tensor):
        total = 0
        for idx in indices:
            i, j = idx[0], idx[1]
            total += tensor[i][j]
        return total

    def sum_tensor_indices_2(self, indices, tensor):
        # Create a tensor with the appropriate dimensions
        j = indices.clone().detach().long().to(self.DEVICE)
        # Select the values using advanced indexing and sum along dim=-1
        row_sums = tensor[:, j].sum(dim=-1)

        # Reshape the result to an L x 1 tensor
        return row_sums.reshape(-1, 1).clone().detach()            
        
        
    def make_table(self, L):
        """
        Make table of all (positive, neutral, negative) charges -> (i, j, k)
        such that: 
            i + j + k = L
            (1 * i) + (0 * j) + (-1 * k) = target_charge

        Arguments:
            L: int
                - length of sequence, defined as seq.shape[0]        
            target_charge : float
                - Target charge for the sequence to be guided towards
                
        Returns:
            table: N x 3 tensor
                - All combinations of i, j, k such that the above conditions are satisfied
        """

        table = []
        for i in range(L):
            for j in range(L):
                for k in range(L):
                    # Check that number of residues = L and that sum of charge (i - k) = target_charge
                    # and that there are no 0 entries, as having no pos, no neg, or no neutral is not realistic
                    if i+j+k == L and i-k == self.target_charge and i != 0 and j != 0 and k != 0:
                        table.append([i,j,k])
        return torch.tensor(np.array(table))
    
    
    def classify_resis(self, seq):
        """
        Classify each position in seq as either positive, neutral, or negative.
        Classification = max( [sum(positive residue logits), sum(neutral residue logits), sum(negative residue logits)] )
        
        Arguments:
            seq: L x 21 tensor
                - sequence logits from the model
        
        Returns: 
            charges: tensor
                - 1 x 3 tensor counting total # of each charge type in the input sequence
                - charges[0] = # positive residues
                - charges[1] = # neutral residues
                - charges[2] = # negative residues                
            charge_classification: tensor
                - L x 1 tensor of each position's classification. 1 is positive, 0 is neutral, -1 is negative
        """
        L = seq.shape[0]
        # Get softmax of seq
        soft_seq = torch.softmax(seq.clone(),dim=-1).requires_grad_(requires_grad=True).to(self.DEVICE)

        # Sum the softmax of all the positive and negative charges along dim = -1 (21 amino acids):
        # Sum across c-term pKs
        sum_cterm_charged = self.sum_tensor_indices(self.cterm_charged_idx, soft_seq).item()
        # print(f'SUM OF CTERM CHARGED RESIS: {sum_cterm_charged}')
        # print(type(sum_cterm_charged.item()))
        sum_cterm_neutral = self.sum_tensor_indices(self.cterm_neutral_idx, soft_seq).item()
        # print(f'SUM OF CTERM NEUTRAL RESIS: {sum_cterm_neutral}')
    
        # Classify c-term as negative or neutral
        cterm_max = max(sum_cterm_charged, sum_cterm_neutral)
        # print(f'CTERM MAX: {cterm_max}')
        if cterm_max == sum_cterm_charged:
            cterm_class = torch.tensor([[-1]]).to(self.DEVICE)
        else:
            cterm_class = torch.tensor([[0]]).to(self.DEVICE)
        # Prep cterm dataframe
        cterm_df = torch.tensor([[0, sum_cterm_neutral, sum_cterm_charged, cterm_max, cterm_class]]).to(self.DEVICE)
        
        # Sum across positive, neutral, and negative pKs
        sum_pos = self.sum_tensor_indices_2(self.pos_pKs_idx, soft_seq[1:L-1, ...]).to(self.DEVICE)
        # print(f'SUM POS: {sum_pos}')
        sum_neg = self.sum_tensor_indices_2(self.neg_pKs_idx, soft_seq[1:L-1, ...]).to(self.DEVICE)
        # print(f'SUM NEG: {sum_neg}')
        sum_neutral = self.sum_tensor_indices_2(self.neutral_pKs_idx, soft_seq[1:L-1, ...]).to(self.DEVICE)
        # print(f'SUM NEUTRAL: {sum_neutral}')  
        
        # Classify non-terminal residues along dim = -1
        middle_max, _ = torch.max(torch.stack((sum_pos, sum_neg, sum_neutral), dim=-1), dim=-1)
        middle_max = middle_max.to(self.DEVICE)
        # create an L x 1 tensor to store the result
        middle_class = torch.zeros((L - 2, 1), dtype=torch.long).to(self.DEVICE)
        # set the values of the result tensor based on which tensor had the maximum value
        middle_class[sum_neg == middle_max] = -1
        middle_class[sum_neutral == middle_max] = 0
        middle_class[sum_pos == middle_max] = 1   
        
        # Prepare df of all middle residue classifications and corresponding values
        middle_df = pd.DataFrame((torch.cat((sum_pos, sum_neutral, sum_neg, middle_max, middle_class), dim=-1)).detach().cpu().numpy())
        middle_df.rename(columns={0: 'sum_pos',
                           1: 'sum_neutral', 2: 'sum_neg', 3: 'middle_max', 4: 'middle_classified'},
                  inplace=True, errors='raise')        
        
        # Sum across n-term pKs
        sum_nterm_charged = self.sum_tensor_indices(self.nterm_charged_idx, soft_seq).to(self.DEVICE)
        # print(f'SUM OF NTERM CHARGED RESIS: {sum_nterm_charged}')        
        sum_nterm_neutral = self.sum_tensor_indices(self.nterm_neutral_idx, soft_seq).to(self.DEVICE)
        # print(f'SUM OF NTERM NEUTRAL RESIS: {sum_nterm_neutral}')
        
        # Classify n-term as negative or neutral
        nterm_max = max(sum_nterm_charged, sum_nterm_neutral)
        if nterm_max == sum_nterm_charged:
            nterm_class = torch.tensor([[-1]]).to(self.DEVICE)
        else:
            nterm_class = torch.tensor([[0]]).to(self.DEVICE)
        nterm_df = torch.tensor([[sum_nterm_charged, sum_nterm_neutral, 0, nterm_max, nterm_class]]).to(self.DEVICE)
        
        # Prep data to be concatenated into output df
        middle_df_2 = (torch.cat((sum_pos, sum_neutral, sum_neg, middle_max, middle_class), dim=-1)).to(self.DEVICE)
        # Concat cterm, middle, and nterm data into one master df with all summed probs, max, and final classification
        full_tens_np = torch.cat((cterm_df, middle_df_2, nterm_df), dim = 0).detach().cpu().numpy()
        classification_df = pd.DataFrame(full_tens_np)
        classification_df.rename(columns={0: 'sum_pos',
                           1: 'sum_neutral', 2: 'sum_neg', 3: 'max', 4: 'classification'},
                  inplace=True, errors='raise') 
        # Count number of positive, neutral, and negative charges that are stored in charge_classification as 1, 0, -1 respectively
        charge_classification = torch.cat((cterm_class, middle_class, nterm_class), dim = 0).to(self.DEVICE)        
        charges = [torch.sum(charge_classification == 1).item(), torch.sum(charge_classification == 0).item(), torch.sum(charge_classification == -1).item()]
        # print('*'*100)
        # print(classification_df)
        
        return torch.tensor(charges), classification_df
    
    def get_target_charge_ratios(self, table, charges):
        """
        Find closest distance between x, y, z in table and i, j, k in charges
        
        Arguments:
            table: N x 3 tensor of all combinations of positive, neutral, and negative charges that obey the conditions in make_table
            charges: 1 x 3 tensor
                - 1 x 3 tensor counting total # of each charge type in the input sequence
                - charges[0] = # positive residues
                - charges[1] = # neutral residues
                - charges[2] = # negative residues

        Returns: 
            target_charge_tensor: tensor
                - 1 x 3 tensor of closest row in table that matches charges of input sequence
        """
        # Compute the difference between the charges and each row of the table
        diff = table - charges

        # Compute the square of the Euclidean distance between the charges and each row of the table
        sq_distance = torch.sum(diff ** 2, dim=-1)

        # Find the index of the row with the smallest distance
        min_idx = torch.argmin(sq_distance)

        # Return the smallest distance and the corresponding row of the table
        target_charge_tensor =  torch.sqrt(sq_distance[min_idx]), table[min_idx]
        #print(f'CLOSEST COMBINATION OF VALID RESIDUES: {target_charge_tensor[1]}')
        return target_charge_tensor[1]
    
    def draft_resis(self, classification_df, target_charge_tensor):
        """
        Based on target_charge_tensor, draft the top (i, j, k) positive, neutral, and negative positions from 
        charge_classification and return the idealized guided_charge_classification. 
        guided_charge_classification will determine whether the gradients should be positive or negative
        
        Draft pick algorithm for determining gradient guided_charge_classification:
            1) Define how many positive, negative, and neutral charges are needed
            2) Current charge being drafted = sign of target charge, otherwise opposite charge
            3) From the classification_df of the currently sampled sequence, choose the position with the highest probability of being current_charge
            4) Make that residue +1, 0, or -1 in guided_charge_classification to dictate the sign of gradients
            5) Keep drafting that residue charge until it is used up, then move to the next type
        
        Arguments:               
            classification_df: tensor
                - L x 1 tensor of each position's classification. 1 is positive, 0 is neutral, -1 is negative                
            target_charge_tensor: tensor
                - 1 x 3 tensor of closest row in table that matches charges of input sequence

        Returns:
            guided_charge_classification: L x 1 tensor
                - L x 1 tensor populated with 1 = positive, 0 = neutral, -1 = negative
                - in get_gradients, multiply the gradients by guided_charge_classification to determine which direction 
                the gradients should guide toward based on the current sequence distribution and the target charge
        """
        charge_dict = {'pos': 0, 'neutral': 0, 'neg': 0}
        # Define the target number of positive, neutral, and negative charges
        charge_dict['pos'] = target_charge_tensor[0].detach().clone()
        charge_dict['neutral'] = target_charge_tensor[1].detach().clone()
        charge_dict['neg'] = target_charge_tensor[2].detach().clone()
        # Determine which charge to start drafting
        if self.target_charge > 0:
            start_charge = 'pos'
        elif self.target_charge < 0:
            start_charge = 'neg'
        else:
            start_charge = 'neutral'

        # Initialize guided_charge_classification
        guided_charge_classification = torch.zeros((classification_df.shape[0], 1))

        # Start drafting
        draft_charge = start_charge        
        while charge_dict[draft_charge] > 0:
            # Find the residue with the max probability for the current draft charge
            max_residue_idx = classification_df.loc[:, ['sum_' + draft_charge]].idxmax()[0]
            # print(max_residue_idx[0])
            # print(type(max_residue_idx))
            #print(f'MAX RESIDUE INDEX for {draft_charge}: {max_residue_idx}')
            # Populate guided_charge_classification with the appropriate charge
            if draft_charge == 'pos':
                guided_charge_classification[max_residue_idx] = 1
            elif draft_charge == 'neg':
                guided_charge_classification[max_residue_idx] = -1
            else:
                guided_charge_classification[max_residue_idx] = 0
            # Remove selected row from classification_df
            classification_df = classification_df.drop(max_residue_idx)
            # print(classification_df)
            # Update charges dictionary
            charge_dict[draft_charge] -= 1
            #print(f'{charge_dict[draft_charge]} {draft_charge} residues left to draft...')
            # Switch to the other charged residue if the starting charge has been depleted
            if charge_dict[draft_charge] == 0:
                if draft_charge == start_charge:
                    draft_charge = 'neg' if start_charge == 'pos' else 'pos'
                elif draft_charge == 'neg':
                    draft_charge = 'pos'
                elif draft_charge == 'pos':
                    draft_charge = 'neg'
                else:
                    draft_charge = 'neutral'

        return guided_charge_classification.requires_grad_()
       
    def get_gradients(self, seq):#, guided_charge_classification):
        """
        Calculate gradients with respect to SEQUENCE CHARGE at pH.
        Uses a MSE loss.

        Arguments
        ---------
        seq : tensor
            L X 21 logits after saving seq_out from xt

        Returns
        -------
        gradients : list of tensors
            gradients of soft_seq with respect to loss on partial_charge
        """        
        # Get softmax of seq
        # soft_seq = torch.softmax(seq.clone(),dim=-1).requires_grad_(requires_grad=True).to(DEVICE)
        soft_seq = torch.softmax(seq,dim=-1).requires_grad_(requires_grad=True).to(DEVICE)

        # Get partial positive charges only for titratable residues
        pos_charge = torch.where(self.pos_pKs_matrix != 0, (1.0 / ((10 ** (self.pH - self.pos_pKs_matrix)) + 1.0)), 0)
        neg_charge = torch.where(self.neg_pKs_matrix != 0, (1.0 / ((10 ** (self.neg_pKs_matrix - self.pH)) + 1.0)), 0)
        # partial_charge = torch.sum((soft_seq*(pos_charge - neg_charge)).requires_grad_(requires_grad=True))
        
        
        if self.loss_type == 'simple':
            # Calculate net partial charge of soft_seq
            partial_charge = torch.sum((soft_seq*(pos_charge - neg_charge)).requires_grad_(requires_grad=True))

            print(f'CURRENT PARTIAL CHARGE: {partial_charge.item()}')
            # Calculate MSE loss on partial_charge
            loss = ((partial_charge - self.target_charge)**2)**0.5
            #print(f'LOSS: {loss}')
            # Take backward step
            loss.backward()            
            # Get gradients from soft_seq
            self.gradients = soft_seq.grad            
            
            # plt.imshow(self.gradients)
            # plt.colorbar()
            # plt.title('gradients')
            
        elif self.loss_type == 'simple2':
            # Calculate net partial charge of soft_seq
            # partial_charge = torch.sum((soft_seq*(pos_charge - neg_charge)).requires_grad_(requires_grad=True))

            print(f'CURRENT PARTIAL CHARGE: {partial_charge.item()}')
            # Calculate MSE loss on partial_charge
            loss = (((torch.sum((soft_seq*(pos_charge - neg_charge)).requires_grad_(requires_grad=True)))
                     - self.target_charge)**2)**0.5
            #print(f'LOSS: {loss}')
            # Take backward step
            loss.backward()            
            # Get gradients from soft_seq
            self.gradients = soft_seq.grad            
            
            # plt.imshow(self.gradients)
            # plt.colorbar()
            # plt.title('gradients')        
            
        elif self.loss_type == 'complex':
            # Preprocessing using method functions
            table = self.make_table(seq.shape[0])
            charges, classification_df = self.classify_resis(seq)
            target_charge_tensor = self.get_target_charge_ratios(table, charges)
            guided_charge_classification = self.draft_resis(classification_df, target_charge_tensor)

            # Calculate net partial charge of soft_seq
            soft_partial_charge = (soft_seq*(pos_charge - neg_charge))
            # print(f'SOFT PARTIAL CHARGE SHAPE: {soft_partial_charge.shape}')
            # Define partial charge as the sum of softmax * partial charge matrix
            partial_charge = torch.sum(soft_partial_charge, dim=-1).requires_grad_()
            #print(partial_charge)
            # partial_charge = torch.sum((soft_seq*(pos_charge - neg_charge)).requires_grad_(requires_grad=True))
            print(f'CURRENT PARTIAL CHARGE: {partial_charge.sum().item()}')            
            
            # print(f'DIFFERENCE BETWEEN TARGET CHARGES AND CURRENT CHARGES: {((guided_charge_classification.to(self.DEVICE) - partial_charge.unsqueeze(1).to(self.DEVICE))**2)**0.5}')
            
            # Calculate loss on partial_charge
            loss = torch.mean(((guided_charge_classification.to(self.DEVICE) - partial_charge.unsqueeze(1).to(self.DEVICE))**2)**0.5)
            # loss = torch.mean((guided_charge_classification.to(self.DEVICE) - partial_charge.to(self.DEVICE))**2)
            #print(f'LOSS: {loss}')
            # Take backward step
            loss.backward()            
            # Get gradients from soft_seq
            self.gradients = soft_seq.grad
            
            # print(f'GUIDED CHARGE CLASSIFICATION SHAPE: {guided_charge_classification.shape}')
            # print(f'PARTIAL CHARGE SHAPE: {partial_charge.unsqueeze(1).shape}')   
            # print(partial_charge)
            # fig, ax = plt.subplots(1,2, dpi=200)
            # ax[0].imshow((partial_charge.unsqueeze(1)).detach().numpy())
            # ax[0].set_title('soft_seq partial charge')
            # ax[1].imshow(self.gradients)#.detach().numpy())
            # ax[1].set_title('gradients')            
            # print(seq)
        return -self.gradients*self.potential_scale

class PSSMbias(Potential):

    def __init__(self, args, features, potential_scale, DEVICE):

        self.features = features
        self.args = args
        self.potential_scale = potential_scale
        self.DEVICE = DEVICE
        self.PSSM = np.loadtxt(args['PSSM'], delimiter=",", dtype=float)
        self.PSSM = torch.from_numpy(self.PSSM).to(self.DEVICE)

    def get_gradients(self, seq):
        print(seq.shape)


        return self.PSSM*self.potential_scale


######################### Feature Potentials ##############################

#From hallucination
def get_entropy_loss(logit_s, mask=None, beta=1, dist_bins=37, eps=1e-16, type_='orig',k=5):
    '''
    net_out - dict with keys: 'c6d' (all pairwise 6D transforms) and/or 'c3d' (3D cartesian coordinates).
                This function uses 'c6d' only. It is a dictionary with the following keys:
                keys: ['p_dist', 'p_omega', 'p_theta', 'p_phi']
                vals: (torch.float32) [batch_size, sequence_length, sequence_length, num_categories]
    mask - 1.0 to include ij pair in cce calculation. The diagonal is automatically excluded.
            (torch.float32)[sequence_length, sequence_length]
    '''

    def entropy_orig(p, mask):
        S_ij = -(p * torch.log(p + eps)).sum(axis=-1)
        S_ave = torch.sum(mask * S_ij, axis=(1,2)) / (torch.sum(mask, axis=(1,2)) + eps)
        return S_ave

    def entropy_leaky(p,p_, mask):
        S_ij = -(p * torch.log(p_ + eps)[...,:-1]).sum(axis=-1)
        S_ave = torch.sum(mask * S_ij, axis=(1,2)) / (torch.sum(mask, axis=(1,2)) + eps)
        return S_ave

    def entropy_leaky_min(p,p_, mask):
        S_ij = -(p * torch.log(p_ + eps)[...,:-1]).sum(axis=-1)
        #pick top k
        S_ij = mask * S_ij
        S_ij[S_ij==0] = 100
        S_ij_min=torch.topk(S_ij, k,largest=False)[0]
        S_ij_min[S_ij_min==100] = 0
        S_ave=torch.sum(S_ij_min,axis=(1,2))/ (torch.sum(mask, axis=(1,2)) + eps)*10 #*k
        return S_ave

    def entropy(p,p_, mask):
        if type_=='orig':
            return entropy_orig(p, mask)
        elif type_=='leaky':
            return entropy_leaky(p,p_, mask)
        elif type_=='leaky_min':
            return entropy_leaky_min(p,p_, mask)


    # This loss function uses c6d
    dict_pred = {'p_dist':torch.softmax(torch.permute(logit_s[0], (0,2,3,1)),dim=-1).float(), 'p_omega':torch.softmax(torch.permute(logit_s[1], (0,2,3,1)),dim=-1).float(), 'p_theta':torch.softmax(torch.permute(logit_s[2], (0,2,3,1)),dim=-1).float(), 'p_phi':torch.softmax(torch.permute(logit_s[3], (0,2,3,1)),dim=-1).float()} #c6d

    # Housekeeping
    device = dict_pred['p_dist'].device
    B = dict_pred['p_dist'].shape[0]

    # Mask includes all ij pairs except the diagonal by default
    if mask is None:
        L_ptn = dict_pred['p_dist'].shape[1]
        mask = torch.ones(L_ptn)

    # Exclude diag
    L_mask = mask.shape[0]
    mask *= (1 - torch.eye(L_mask, dtype=torch.float32, device=mask.device))

    # Add batch
    mask = mask[None]
    mask = mask.to(device)

    # Modulate sharpness of probability distribution
    # Also exclude >20A bin, which improves output quality
    pd = torch.softmax(torch.log(beta * dict_pred['p_dist'][...,:-1] + eps), axis = -1)
    pd_= torch.softmax(torch.log(beta * dict_pred['p_dist'] + eps), axis = -1)
    po = torch.softmax(torch.log(beta * dict_pred['p_omega'][...,:36] + eps), axis = -1)
    po_= torch.softmax(torch.log(beta * dict_pred['p_omega'] + eps), axis = -1)
    pt = torch.softmax(torch.log(beta * dict_pred['p_theta'][...,:36] + eps), axis = -1)
    pt_= torch.softmax(torch.log(beta * dict_pred['p_theta'] + eps), axis = -1)
    pp = torch.softmax(torch.log(beta * dict_pred['p_phi'][...,:18] + eps), axis = -1)
    pp_= torch.softmax(torch.log(beta * dict_pred['p_phi'] + eps), axis = -1)

    # Entropy loss
    S_d = entropy(pd.to(device),pd_.to(device), mask)
    S_o = entropy(po.to(device),po_.to(device), mask)
    S_t = entropy(pt.to(device),pt_.to(device), mask)
    S_p = entropy(pp.to(device),pp_.to(device), mask)
    loss = 0.25 * (S_d + S_o + S_t + S_p)
    return loss


class feature_Potential:
    
    def get_gradients(features):
        '''
            EVERY POTENTIAL CLASS MUST RETURN GRADIENTS
        '''
        
        sys.exit('ERROR POTENTIAL HAS NOT BEEN IMPLEMENTED')

class Termini_Contacts(feature_Potential):
    
    def __init__(self, args, features, scale, DEVICE):
        self.DEVICE=DEVICE
        self.scale = scale
        self.binder_len = args['binder_len']

    def get_gradients(self, features):
        grads={}
        torch.set_grad_enabled(True)
        L=self.binder_len
        mask=torch.zeros(L,L)
        for i in range(L):
            for j in range(L):
                if i<20 and j>L-20:
                    mask[i,j]=1.
                if j<20 and i>L-20:
                    mask[i,j]=1.
        for prefix in ['','b_']:
            logit_s = [x[:,:,:L,:L] for x in features[f'{prefix}logit_s']]
            # loss = get_entropy_loss(logit_s, mask=mask, beta=1, dist_bins=37, eps=1e-16, type_='leaky',k=5)
            distmap=(torch.stack([torch.ones_like(logit_s[0][0,0]) * i**2 for i in range(0,logit_s[0][0].shape[0])])*torch.softmax(logit_s[0][0], dim=0)).sum(dim=0)
            loss=torch.mean(distmap*mask.to(distmap.device))
            print(loss)
            torch.cuda.empty_cache()
            loss.backward()
            grad=features[f'{prefix}seq_diffused'].grad[0,:,:21]
            grad=grad-grad.min(dim=-1)[0].unsqueeze(-1).expand(-1,21)
            grad=grad/grad.max(dim=-1)[0].unsqueeze(-1).expand(-1,21)
            grad=grad-0.5
            grads[f'{prefix}seq_out'] = grad
            del grad
            # grads[f'{prefix}seq_out'] = torch.autograd.grad(outputs=loss, inputs=features[f'{prefix}seq_diffused'])[0,:,:21]
            del loss
            torch.cuda.empty_cache()
        return {k:v*self.scale for k,v in grads.items()}
    #def get_gradientsV(self, features):

    #    xyz=features['xyz']
    #    b_xyz=features['b_xyz']
    #    
    #    n=torch.mean(xyz[...,0:25,:,:],dim=(-2,-3))
    #    c=torch.mean(xyz[...,:self.binder_len,:,:][...,-25:,:,:],dim=(-2,-3))
    #    b_n=torch.mean(b_xyz[...,0:25,:,:],dim=(-2,-3))
    #    b_c=torch.mean(b_xyz[...,-25:,:,:],dim=(-2,-3))
    #    
    #    
    #    
    #    dist=torch.sqrt(torch.square(n[...,0]-c[...,0])+torch.square(n[...,1]-c[...,1])+torch.square(n[...,2]-c[...,2]))
    #    b_dist=torch.sqrt(torch.square(b_n[...,0]-b_c[...,0])+torch.square(b_n[...,1]-b_c[...,1])+torch.square(b_n[...,2]-b_c[...,2]))
    #    
    #    loss = torch.absolute(dist-b_dist)
    #    
    #    loss.backward()
    #    print(loss)
    #    out = {'seq_out':-features['msa_masked'].grad[0,0,0,:,:21]*self.scale, 'b_seq_out':-features['b_msa_masked'].grad[0,0,0,:,:21]*self.scale}
    #    del loss
    #    return out
#    def get_gradients(self, features):

        #xyz=features['xyz']
        #xyz.retain_grad()
        #
        #n=torch.mean(xyz[...,0:25,:,:],dim=(-2,-3))
        #c=torch.mean(xyz[...,:self.binder_len,:,:][...,-25:,:,:],dim=(-2,-3))
        #
        #dist=torch.sqrt(torch.square(n[...,0]-c[...,0])+torch.square(n[...,1]-c[...,1])+torch.square(n[...,2]-c[...,2]))
        #
        #dist.backward()
      
        #grad = features['msa_masked'].grad[0,0,0,:,:21]
        #xyzgrad = xyz.grad
        #print(dist)
        #del dist
        #b_xyz=features['b_xyz']
        #b_xyz.retain_grad()
        #
        #b_n=torch.mean(b_xyz[...,0:25,:,:],dim=(-2,-3))
        #b_c=torch.mean(b_xyz[...,:self.binder_len,:,:][...,-25:,:,:],dim=(-2,-3))
        #
        #b_dist=torch.sqrt(torch.square(b_n[...,0]-b_c[...,0])+torch.square(b_n[...,1]-b_c[...,1])+torch.square(b_n[...,2]-b_c[...,2]))
        #
        #b_dist.backward()

        #b_grad = features['b_msa_masked'].grad[0,0,0,:,:21]
        #b_xyzgrad = b_xyz.grad
        #print(b_dist) 
        #del b_dist
        #return {'seq_out':torch.pow(torch.absolute(grad), 1/4)*torch.sign(grad)*-self.scale}#, 'b_seq_out':torch.pow(torch.absolute(b_grad), 1/4)*torch.sign(b_grad)*-self.scale, 'xyz':xyzgrad*self.scale, 'b_xyz':-b_xyzgrad*self.scale}
            


### ADD NEW POTENTIALS INTO LIST DOWN BELOW ###
POTENTIALS = {'aa_bias':AACompositionalBias, 'charge':ChargeBias, 'hydrophobic':HydrophobicBias, 'PSSM':PSSMbias}
FEATURE_POTENTIALS = {'term_contacts':Termini_Contacts}
