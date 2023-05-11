# utility functions for dealing with contigs during hallucination
import numpy as np
import random, copy, torch, geometry, os, sys
from kinematics import xyz_to_t2d

def parse_range_string(el):
  ''' Splits string with integer or integer range into start and end ints. '''
  if '-' in el:
    s,e = el.split('-')
    s,e = int(s), int(e)
  else:
    s,e = int(el), int(el)
  return s,e

def ranges_to_indexes(range_string):
    '''Converts a string containig comma-separated numeric ranges to a list of integers'''
    idx = []
    for x in range_string.split(','):
        start, end = parse_range_string(x)
        idx.extend(np.arange(start, end+1))
    return np.array(idx)

def parse_contigs(contig_input, pdb_id):
  '''
  Input: contig start/end by pdb chain and residue number as in the pdb file
         ex - B12-17
  Output: corresponding start/end indices of the "features" numpy array (idx0)
  '''
  contigs = []
  for con in contig_input.split(','):
    pdb_ch = con[0]
    pdb_s, pdb_e = parse_range_string(con[1:])
    
    np_s = pdb_id.index((pdb_ch, pdb_s))
    np_e = pdb_id.index((pdb_ch, pdb_e))
    
    contigs.append([np_s, np_e])
  return contigs


def mk_feat_hal_and_mappings(hal_2_ref_idx0, pdb_out):
  #####################################
  # rearrange ref features according to hal_2_ref_idx0
  #####################################
  #1. find corresponding idx0 in hal and ref
  hal_idx0 = []
  ref_idx0 = []
  
  for hal, ref in enumerate(hal_2_ref_idx0):
    if ref is not None:
      hal_idx0.append(hal)
      ref_idx0.append(ref)
      
  hal_idx0 = np.array(hal_idx0, dtype=int)
  ref_idx0 = np.array(ref_idx0, dtype=int)
        
  #2. rearrange the 6D features
  hal_len = len(hal_2_ref_idx0)
  if 'feat' in pdb_out:
      d_feat = pdb_out['feat'].shape[3:]
      
      feat_hal = np.zeros((1, hal_len, hal_len) + d_feat)
      feat_ref = pdb_out['feat']  # (B,L,L,...)
    
      feat_hal[:, hal_idx0[:,None], hal_idx0[None,:]] = feat_ref[:, ref_idx0[:,None], ref_idx0[None,:]]
  else:
    feat_hal = None
      
  #3. make the 1d binary mask, for backwards compatibility
  hal_2_ref_idx0 = np.array(hal_2_ref_idx0, dtype=np.float32)  # convert None to NaN
  mask_1d = (~np.isnan(hal_2_ref_idx0)).astype(float)
  mask_1d = mask_1d[None]
  
  
  #####################################
  # mappings between hal and ref
  #####################################
  mappings = {
    'con_hal_idx0': hal_idx0.tolist(),
    'con_ref_idx0': ref_idx0.tolist(),
    'con_hal_pdb_idx': [('A',i+1) for i in hal_idx0],
    'con_ref_pdb_idx': [pdb_out['pdb_idx'][i] for i in ref_idx0],
    'mask_1d': mask_1d,
  }
  
  return feat_hal, mappings

def scatter_feats(template_mask, feat_1d_ref=None, feat_2d_ref=None, pdb_idx=None):
  '''
  Scatters 1D and/or 2D reference features according to mappings in hal_2_ref_idx0
  
  Inputs
  ----------
  hal_2_ref_idx0: (list; length=L_hal)
      List mapping hal_idx0 positions to ref_idx0 positions.
      "None" used for indices that do not map to ref.
      ex: [None, None, 3, 4, 5, None, None, None, 34, 35, 36]
  feat_1d_ref: (np.array; (batch, L_ref, ...))
      1D refence features to scatter
  feat_1d_ref: (np.array; (batch, L_ref, L_ref, ...))
  pdb_idx: (list)
      List of pdb chain and residue numbers, in the order that pdb features were read/parsed.
  
  Outputs
  ----------
  feat_1d_hal: (np.array, (batch, L_hal, ...))
      Scattered 1d reference features. "None" mappings are 0.
  feat_2d_hal: (np.array, (batch, L_hal, L_hal, ...))
      Scattered 2d reference features. "None" mappings are 0.
  mappings: (dict)
      Keeps track of corresponding possitions in ref and hal proteins.
  '''
  hal_2_ref_idx0, _ = contigs.sample_mask(template_mask, pdb_idx)
  out = {}
  
  # Find corresponding idx0 in hal and ref
  hal_idx0 = []
  ref_idx0 = []
  hal_len = len(hal_2_ref_idx0)
  
  for hal, ref in enumerate(hal_2_ref_idx0):
    if ref is not None:
      hal_idx0.append(hal)
      ref_idx0.append(ref)
      
  hal_idx0 = np.array(hal_idx0, dtype=int)
  ref_idx0 = np.array(ref_idx0, dtype=int)
  
  # Make the 1d binary mask, for backwards compatibility
  hal_2_ref_idx0 = np.array(hal_2_ref_idx0, dtype=np.float32)  # convert None to NaN
  mask_1d = (~np.isnan(hal_2_ref_idx0)).astype(float)
  mask_1d = mask_1d[None]
        
  # scatter 2D features
  if feat_2d_ref is not None:
      B = feat_2d_ref.shape[0]
      d_feat = feat_2d_ref.shape[3:]
      feat_2d_hal = np.zeros((B, hal_len, hal_len)+d_feat)
      feat_2d_hal[:, hal_idx0[:,None], hal_idx0[None,:]] = feat_2d_ref[:, ref_idx0[:,None], ref_idx0[None,:]]
      out['feat_2d_hal'] = feat_2d_hal
      
  # scatter 1D features
  if feat_1d_ref is not None:
      B = feat_1d_ref.shape[0]
      d_feat = feat_1d_ref.shape[2:]
      feat_1d_hal = np.zeros((B, hal_len)+d_feat)
      feat_1d_hal[:, hal_idx0] = feat_1d_ref[:, ref_idx0]
      out['feat_1d_hal'] = feat_1d_hal
  
  # Mappings between hal and ref
  mappings = {
      'con_hal_idx0': hal_idx0.tolist(),
      'con_ref_idx0': ref_idx0.tolist(),
      'mask_1d': mask_1d,
  }
  
  if pdb_idx is not None:
      mappings.update({
          'con_hal_pdb_idx': [('A',i+1) for i in hal_idx0],
          'con_ref_pdb_idx': [pdb_idx[i] for i in ref_idx0],
      })
      
  out['mappings'] = mappings
  
  return out

def scatter_contigs(contigs, pdb_out, L_range, keep_order=False, min_gap=0):
  '''
  Randomly places contigs in a protein within the length range.
  
  Inputs
    Contig: A continuous range of residues from the pdb.
            Inclusive of the begining and end
            Must start with the chain number. Comma separated
            ex: B6-11,A12-19
    pdb_out: dictionary from the prep_input function
    L_range: String range of possible lengths.
              ex: 90-110
              ex: 70
    keep_order: keep contigs in the provided order or randomly permute
    min_gap: minimum number of amino acids separating contigs
    
  Outputs
    feat_hal: target pdb features to hallucinate
    mappings: dictionary of ways to convert from the hallucinated protein
              to the reference protein  
  
  '''
  
  ref_pdb_2_idx0 = {pdb_idx:i for i, pdb_idx in enumerate(pdb_out['pdb_idx'])}
  
  #####################################
  # make a map from hal_idx0 to ref_idx0. Has None for gap regions
  #####################################
  #1. Permute contig order
  contigs = contigs.split(',')
  
  if not keep_order:
    random.shuffle(contigs)
    
  #2. convert to ref_idx0
  contigs_ref_idx0 = []
  for con in contigs:
    chain = con[0]
    s, e = parse_range_string(con[1:])
    contigs_ref_idx0.append( [ref_pdb_2_idx0[(chain, i)] for i in range(s, e+1)] )
  
  #3. Add minimum gap size
  for i in range(len(contigs_ref_idx0) - 1):
    contigs_ref_idx0[i] += [None] * min_gap
    
  #4. Sample protein length
  L_low, L_high = parse_range_string(L_range)
  L_hal = np.random.randint(L_low, L_high+1)
  
  L_con = 0
  for con in contigs_ref_idx0:
    L_con += len(con)
    
  L_gaps = L_hal - L_con
  
  if L_gaps <= 1:
    print("Error: The protein isn't long enough to incorporate all the contigs."
          "Consider reduce the min_gap or increasing L_range")
    return
  
  #5. Randomly insert contigs into gaps
  hal_2_ref_idx0 = np.array([None] * L_gaps, dtype=float)  # inserting contigs into this
  n_contigs = len(contigs_ref_idx0)  
  insertion_idxs = np.random.randint(L_gaps + 1, size=n_contigs)
  insertion_idxs.sort()
  
  for idx, con in zip(insertion_idxs[::-1], contigs_ref_idx0[::-1]):
    hal_2_ref_idx0 = np.insert(hal_2_ref_idx0, idx, con)
    
  #6. Convert mask to feat_hal and mappings
  hal_2_ref_idx0 = [int(el) if ~np.isnan(el) else None for el in hal_2_ref_idx0]  # convert nan to None
  feat_hal, mappings = mk_feat_hal_and_mappings(hal_2_ref_idx0, pdb_out)
  
  #7. Generate str of the sampled mask
  contig_positive = np.array(hal_2_ref_idx0) != None
  boundaries = np.where(np.diff(contig_positive))[0]
  start_idx0 = np.concatenate([np.array([0]), boundaries+1])
  end_idx0 = np.concatenate([boundaries, np.array([contig_positive.shape[0]])-1])
  lengths = end_idx0 - start_idx0 + 1
  is_contig = contig_positive[start_idx0]

  sampled_mask = []
  con_counter = 0

  for i, is_con in enumerate(is_contig):
    if is_con:
      sampled_mask.append(contigs[con_counter])
      con_counter += 1
    else:
      len_gap = lengths[i]
      sampled_mask.append(f'{len_gap}-{len_gap}')

  sampled_mask = ','.join(sampled_mask)
  mappings['sampled_mask'] = sampled_mask
  
  return feat_hal, mappings

def get_receptor_contig(ref_pdb_idx):
  rec_pdb_idx = [idx for idx in ref_pdb_idx if idx[0]=='R']
  return SampledMask.contract(rec_pdb_idx)

def mk_con_to_set(mask, set_id=None, args=None, ref_pdb_idx=None):
  '''
  Maps a mask or list of contigs to a set_id. If no set_id is provided, it treats
  everything as set 0.
  
  Input
  -----------
  mask (str): Mask or list of contigs. Ex: 3,B6-11,12,A12-19,9 or Ex: B6-11,A12-19
  ref_pdb_idx (List(ch, res)): pdb idxs of the reference pdb. Ex: [(A, 2), (A, 3), ...]
  args: Arguments object. Must have args.receptor
  set_id (list): List of integers. Length must match contigs in mask. Ex: [0,1]
  
  Output
  -----------
  con_to_set (dict): Maps str of contig to integer
  '''
  
  # Extract contigs
  cons = [l for l in mask.split(',') if l[0].isalpha()]
  
  # Assign all contigs to set 0 if set_id is not passed
  if set_id is None:
    set_id = [0] * len(cons)
    
  con_to_set = dict(zip(cons, set_id))
  
  # Assign receptor to set 0
  if args.receptor:
    receptor_contig = get_receptor_contig(ref_pdb_idx)
    con_to_set.update({receptor_contig: 0})
  
  return con_to_set

def parse_range(_range):
  if '-' in _range:
    s, e = _range.split('-')
  else:
    s, e = _range, _range

  return int(s), int(e)

def parse_contig(contig):
  '''
  Return the chain, start and end residue in a contig or gap str.
  
  Ex:
  'A4-8' --> 'A', 4, 8
  'A5'   --> 'A', 5, 5
  '4-8'  --> None, 4, 8
  'A'    --> 'A', None, None
  '''
  
  # is contig
  if contig[0].isalpha():
    ch = contig[0]
    if len(contig) > 1:
      s, e = parse_range(contig[1:])
    else:
      s, e = None, None
  # is gap
  else:
    ch = None
    s, e = parse_range(contig)
      
  return ch, s, e
  
def mask_as_list(sampled_mask):
  '''
  Make a length L_hal list, with each position pointing to a ref_pdb_idx (or None)
  '''
  mask_list = []
  for l in sampled_mask.split(','):
    ch, s, e = parse_contig(l)
    # contig
    if ch is not None:  
      mask_list += [(ch, idx) for idx in range(s, e+1)]
    # gap
    else:
      mask_list += [None for _ in range(s, e+1)]
      
  return mask_list

def mask_subset(sampled_mask, subset):
  '''
  Returns a 1D boolean array of where a subset of the contig is in the hallucinated protein
  
  Input
  ---------
  subset (str): Some chain and residue subset of the contigs. Ex: A10-15
      Can also just pass chain. All contig residues from that chain are selected. Ex: R
  
  Ouput
  ---------
  m_1d (np.array): Boolean array where subset appears in the hallucinated protein
  
  '''
  mask_list = mask_as_list(sampled_mask)
  m_1d = []
  
  ch_subset, s, e = parse_contig(subset)
  assert ch_subset.isalpha(), '"Subset" must include a chain reference'
  
  if (s is None) or (e is None):
    s = -np.inf
    e = np.inf
    
  for l in mask_list:
    if l is None:
      continue
      
    ch, idx = l
    if (ch == ch_subset) and (idx >= s) and (idx <= e):
      m_1d.append(True)
    else:
      m_1d.append(False)
  
  return np.array(m_1d)  

def mk_cce_and_hal_mask_2d(sampled_mask, con_to_set=None):
  '''
  Makes masks for ij pixels where the cce and hallucination loss should be applied.

  Inputs
  ---------------
  sampled_mask (str): String of where contigs should be applied. Ex: 3,B6-11,12,A12-19,9
  cce_cutoff (float): Apply cce loss to cb-cb distances less than this value. Angstroms.
  con_to_set (dict): Dictionary mapping the string of a contig (ex: 'B6-11') to an integer.
  L_rec (int): Length of the receptor, if hallucinating in the context of the receptor.
  
  Outputs
  ---------------
  mask_cce (np.array, (L_hal, L_hal)): Boolean array. True where cce loss should be applied. 
  mask_hal (np.array, (L_hal, L_hal)): Boolean array. True where hallucination loss should be applied. 
  '''
  if con_to_set is None:
    con_to_set = mk_con_to_set(sampled_mask)

  # Length of hallucinated protein
  L_hal, L_max = mask_len(sampled_mask)
  assert L_hal == L_max, 'A sampled mask must have gaps of a single length.'

  # Map each contig to a 1D boolean mask
  m_con = dict()
  start_idx = 0
  for l in sampled_mask.split(','):
    if l[0].isalpha():
      s, e = parse_range_string(l[1:])
      L_con = e - s + 1
      m = np.zeros(L_hal, dtype=bool)
      m[start_idx:start_idx+L_con] = True

      m_con[l] = m
      start_idx += L_con
    else:
      L_gap, _ = parse_range_string(l)
      start_idx += L_gap

  # Combine contigs masks from each set to make 2D mask
  mask_cce = np.zeros((L_hal, L_hal), dtype=bool)
  for set_id in set(con_to_set.values()):    
    # gather all masks from contigs in the same set
    masks = [m_con[k] for k,v in con_to_set.items() if v == set_id]
    mask_1D = np.any(masks, axis=0)    
    update = mask_1D[:,None] * mask_1D[None,:]
    mask_cce = np.any([mask_cce, update], axis=0)
    
  # Make mask_hal
  mask_hal = ~mask_cce
    
  # Don't apply ANY losses on diagonal
  mask_cce[np.arange(L_hal), np.arange(L_hal)] = False
  mask_hal[np.arange(L_hal), np.arange(L_hal)] = False
    
  # Don't apply ANY losses to receptor
  m_1d_rec = mask_subset(sampled_mask, 'R')
  m_2d_rec = m_1d_rec[:, None] * m_1d_rec[None, :]
  mask_cce *= ~m_2d_rec
  mask_hal *= ~m_2d_rec
    
  return mask_cce, mask_hal
  

def apply_mask(mask, pdb_out):
  '''
  Uniformly samples gap lengths, then gathers the ref features
  into the target hal features
  
  Inputs
  --------------
  mask: specify the order and ranges of contigs and gaps
        Contig - A continuous range of residues from the pdb.
                Inclusive of the begining and end
                Must start with the chain number
                ex: B6-11
        Gap - a gap length or a range of gaps lengths the 
                model is free to hallucinate
                Gap ranges are inclusive of the end
                ex: 9-21

        ex - '3,B6-11,9-21,A36-42,20-30,A12-24,3-6'
  
  pdb_out: dictionary from the prep_input function
  
  
  Outputs
  -------------
  feat_hal: features from pdb_out scattered according to the sampled mask
  mappings: dict keeping track of corresponding positions in the ref and hal features
  
  '''
  
  ref_pdb_2_idx0 = {pdb_idx:i for i, pdb_idx in enumerate(pdb_out['pdb_idx'])}
  
  #1. make a map from hal_idx0 to ref_idx0. Has None for gap regions
  hal_2_ref_idx0 = []
  sampled_mask = []
  for el in mask.split(','):

    if el[0].isalpha():  # el is a contig
      sampled_mask.append(el)
      chain = el[0]
      s,e = parse_range_string(el[1:])
      
      for i in range(s, e+1):
        idx0 = ref_pdb_2_idx0[(chain, i)]
        hal_2_ref_idx0.append(idx0)
        
    else:  # el is a gap
      # sample gap length
      s,e = parse_range_string(el)
      gap_len = np.random.randint(s, e+1)
      hal_2_ref_idx0 += [None]*gap_len
      sampled_mask.append(f'{gap_len}-{gap_len}')
      
  #2. Convert mask to feat_hal and mappings 
  feat_hal, mappings = mk_feat_hal_and_mappings(hal_2_ref_idx0, pdb_out)
    
  #3. Record the mask that was sampled
  mappings['sampled_mask'] = ','.join(sampled_mask)
  
  return feat_hal, mappings


def sample_mask(mask, pdb_idx):
  '''
  Uniformly samples gap lengths, then gathers the ref features
  into the target hal features
  
  Inputs
  --------------
  mask: specify the order and ranges of contigs and gaps
        Contig - A continuous range of residues from the pdb.
                Inclusive of the begining and end
                Must start with the chain number
                ex: B6-11
        Gap - a gap length or a range of gaps lengths the 
                model is free to hallucinate
                Gap ranges are inclusive of the end
                ex: 9-21

        ex - '3,B6-11,9-21,A36-42,20-30,A12-24,3-6'  
  
  Outputs
  -------------
  hal_2_ref_idx0: (list; length=L_hal)
      List mapping hal_idx0 positions to ref_idx0 positions.
      "None" used for indices that do not map to ref.
      ex: [None, None, 3, 4, 5, None, None, None, 34, 35, 36]
  sampled_mask: (str)
      string of the sampled mask, so the transformations can be reapplied
      ex - '3-3,B6-11,9-9,A36-42,20-20,A12-24,5-5'  
  
  '''
  
  ref_pdb_2_idx0 = {pdb_i:i for i, pdb_i in enumerate(pdb_idx)}
  
  #1. make a map from hal_idx0 to ref_idx0. Has None for gap regions
  hal_2_ref_idx0 = []
  sampled_mask = []
  for el in mask.split(','):

    if el[0].isalpha():  # el is a contig
      sampled_mask.append(el)
      chain = el[0]
      s,e = parse_range_string(el[1:])
      
      for i in range(s, e+1):
        idx0 = ref_pdb_2_idx0[(chain, i)]
        hal_2_ref_idx0.append(idx0)
        
    else:  # el is a gap
      # sample gap length
      s,e = parse_range_string(el)
      gap_len = np.random.randint(s, e+1)
      hal_2_ref_idx0 += [None]*gap_len
      sampled_mask.append(f'{gap_len}-{gap_len}')
  
  return hal_2_ref_idx0, sampled_mask


class GapResampler():
  def __init__(self, use_bkg=True):
    '''

    '''

    self.counts_passed = {}  # dictionary for tallying counts of gap lengths for designs passing some threshold
    self.counts_bkg = {}
    self.use_bkg = use_bkg
    
    
  def clean_mask(self, mask):
    '''
    Makes mask into a cononical form.
    Ensures masks always alternate gap, contig and that 
    masks begin and end with a gap (even of length 0)
    
    Input
    -----------
    masks: list of masks (str). Mask format: comma separted list
        of alternating gap_length (int or int-int), contig.
        Ex - 9,A12-19,15,B45-52 OR 9-9,A12-19,15-15,B45-52
        
    Output
    -----------
    A canonicalized mask. Ex: N,9,A12-19,15,B45-52,0,C
    '''
    mask = mask.split(',')
    mask_out = []
    was_contig = True
    was_gap = False

    for i, el in enumerate(mask):
      is_contig = el[0].isalpha()
      is_gap = not is_contig
      is_last = i == len(mask) - 1
      
      # accepting gaps as either x-x or just x
      if is_gap:
        if '-' in el:
          x1, x2 = el.split('-')
          if x1 != x2:
            print(f"Error: Gap must not be a range: {mask}")
            return None
          gap = x1
        else:
          gap = el

      if is_contig: 
        contig = el

      # gap -> contig: just append new contig
      if (was_gap and is_contig):
        mask_out.append(contig)

      # contig -> gap: just append gap
      elif (was_contig and is_gap):
        mask_out.append(gap)

      # contig -> contig: insert gap of 0, then add contig
      elif (was_contig and is_contig):
        mask_out.append('0')
        mask_out.append(contig)

      # gap -> gap: add them
      elif (was_gap and is_gap):
        combined_len = int(mask_out[-1]) + int(gap)
        mask_out[-1] = str(combined_len)

      # ensure last mask element is a gap
      if (is_last and is_contig):
        mask_out.append('0')

      # update what previous element was
      was_contig = el[0].isalpha()
      was_gap = ~is_contig
      
    # add 'N' and 'C' contigs
    mask_out.insert(0, 'N')
    mask_out.append('C')
    
    return ','.join(mask_out)

  
  def add_mask(self, mask, counting_dict):
    '''
    Adds counts of gap lengths to counting_dict
    
    Inputs
    -----------
    masks: list of masks (str). Mask format: comma separted list
        of alternating gap_length (int or int-int), contig.
        Ex - 9,A12-19,15,B45-52 OR 9-9,A12-19,15-15,B45-52
    '''
    mask = self.clean_mask(mask)
    mask = mask.split(',')
    n_gaps = len(mask) // 2
    
    # count occurances of contig,gap,contig triples
    for i in range(n_gaps):
      con1, gap, con2 = mask[2*i : 2*i+3]
      
      # count gap length
      if con1 in counting_dict:
        if (gap, con2) in counting_dict[con1]:
          counting_dict[con1][(gap, con2)] += 1
        else:
          counting_dict[con1][(gap, con2)] = 1
      else:
        counting_dict[con1] = {(gap, con2): 1}
        
  
  def add_mask_pass(self, mask):
    '''
    Add a mask that passed to self.counts_passed
    '''
    self.add_mask(mask, self.counts_passed)
    
    
  def add_mask_bkg(self, mask):
    '''
    Add a mask that passed to self.counts_bkg
    '''
    self.add_mask(mask, self.counts_bkg)
      
  
  def get_enrichment(self):
    '''
    Calculate the ratio of counts_passed / count_bkg
    Also notes all contigs
    '''
    if self.use_bkg is False:
      print('Please pass in background masks and set self.use_bkg=True')
      return    
    
    self.counts_enrich = copy.copy(self.counts_passed)
    self.con_all = set()
    
    for con1 in self.counts_enrich.keys():
      self.con_all |= set([con1])
      
      for gap, con2 in self.counts_enrich[con1].keys():
        self.con_all |= set([con2])
        bkg = self.counts_bkg[con1][(gap, con2)]
        cnt = self.counts_passed[con1][(gap, con2)]
        self.counts_enrich[con1][(gap, con2)] = cnt / bkg
        
  def sample_mask(self):
    '''
    Sample a mask
    '''
    searching = True
    while searching:
      n_gaps = len(self.con_all) - 1
      mask = ['N']

      if self.use_bkg:
        counts = self.counts_enrich
      else:
        counts = self.counts_passed
    
      for i in range(n_gaps):
        con_last = mask[-1]
        
        # only allow jump to C as last option
        if i == n_gaps - 1:
          con_used = set(mask[::2])
        else:
          con_used = set(mask[::2]+['C'])
          
        con_free = self.con_all - con_used

        # get available "jumps" (con -> gap, con) you can make
        jumps_all = counts[con_last]
        jumps_free = {k:v for k,v in jumps_all.items() if k[1] in con_free}

        if len(jumps_free) == 0:
          print('No available jumps to continue the mask. Sampling again...')
        else:
          # normalize counts and sample move
          mvs, cnt = zip(*jumps_free.items())
          cnt = np.array(cnt)
          prob = cnt / cnt.sum()
          idx = np.random.choice(len(prob), p=prob)
          mv = mvs[idx]

          # add to the mask
          mask.append(mv[0])
          mask.append(mv[1])
      
        # check that mask has the right number of elements
        if len(mask) == 2*n_gaps + 1:
          searching = False
        else:
          searching = True
      
    return ','.join(mask[1:-1])
  
  
  def gaps_as_ranges(self, mask):
    '''
    Convert gaps of a single int to ranges, for
    backwards compatibility reasons
    '''
    
    mask_out = []
    for el in mask.split(','):
      if el[0].isalpha():
        mask_out.append(el)
      else:
        mask_out.append(f'{el}-{el}')
        
    return ','.join(mask_out)
      
      
def recover_mask(trb):
  '''
  Recover the string of the sampled mask given the trb file
  '''

  L_hal = trb['mask_contig'].shape[0]
  mask = []
  
  for idx0 in range(L_hal):
    # what is the current idx
    if idx0 in trb['con_hal_idx0']:
      is_con = True
      is_gap = False
    else:
      is_con = False
      is_gap = True

    # dealing with the first entry
    if idx0 == 0:
      if is_gap:
        L_gap = 1
      elif is_con:
        ch, idx = trb['con_ref_pdb_idx'][ trb['con_hal_idx0'].tolist().index(idx0) ]
        con_start = f'{ch}{idx}'
        
    # take action based on what happend last time
    else:
      if (was_gap) and (is_gap):
        L_gap +=1
      #elif (was_con) and (is_con):
      #  continue
      elif (was_gap) and (is_con):
        # end gap
        mask.append(f'{L_gap}-{L_gap}')
        # start con
        ch, idx = trb['con_ref_pdb_idx'][ trb['con_hal_idx0'].tolist().index(idx0) ]
        con_start = f'{ch}{idx}'
      elif (was_con) and (is_gap):
        # end con
        ch, idx = trb['con_ref_pdb_idx'][ trb['con_hal_idx0'].tolist().index(idx0) ]
        mask.append(f'{con_start}-{idx}')
        # start gap
        L_gap = 1
        
    # dealing with last entry
    if idx0 == L_hal-1:
      if is_gap:
        mask.append(f'{L_gap}-{L_gap}')
      elif is_con:  # (edge case not handled: con starts and ends on last idx)
        ch, idx = trb['con_ref_pdb_idx'][ trb['con_hal_idx0'].tolist().index(idx0-1) ]
        mask.append(f'{con_start}-{idx}')

    # update what last position was
    was_con = copy.copy(is_con)
    was_gap = copy.copy(is_gap)

  return ','.join(mask)


def mask_len(mask):
  '''
  Calculate the min and max possible length that can 
  be sampled given a mask
  '''
  L_min = 0
  L_max = 0
  
  for el in mask.split(','):
    if el[0].isalpha():  # is con
      con_s, con_e = el[1:].split('-')
      con_s, con_e = int(con_s), int(con_e)
      L_con = con_e - con_s + 1
      L_min += L_con
      L_max += L_con
    
    else:  # is gap
      if '-' in el:
        gap_min, gap_max = el.split('-')
        gap_min, gap_max = int(gap_min), int(gap_max)
        L_min += gap_min
        L_max += gap_max
      else:
        L_min += int(el)
        L_max += int(el)
        
  return L_min, L_max

class SampledMask():
  def __init__(self, mask_str, ref_pdb_idx, con_to_set=None):    
    self.str = mask_str
    self.L_hal = len(self)
    self.L_ref = len(ref_pdb_idx)

    #################
    # con indices in hal and ref
    #################
    self.ref_pdb_idx = ref_pdb_idx
    self.hal_pdb_idx = [('A', i) for i in range(1, len(self)+1)]
    
    hal_idx0 = 0
    con_ref_pdb_idx = []
    con_hal_pdb_idx = []
    con_ref_idx0 = []
    con_hal_idx0 = []
    
    for l in mask_str.split(','):
      ch, s, e = SampledMask.parse_contig(l)
      
      # contig
      if ch:
        for res in range(s, e+1):
          con_ref_pdb_idx.append((ch, res))
          con_hal_pdb_idx.append(('A', hal_idx0+1))
          con_ref_idx0.append(self.ref_pdb_idx.index((ch, res)))
          con_hal_idx0.append(hal_idx0)
          hal_idx0 += 1
      # gap
      else:
        for _ in range(s):
          hal_idx0 += 1
    
    self.con_mappings = {
      'ref_pdb_idx': con_ref_pdb_idx,
      'hal_pdb_idx': con_hal_pdb_idx,
      'ref_idx0': con_ref_idx0,
      'hal_idx0': con_hal_idx0,      
    }
    
    #################
    # con_to_set mapping
    #################
    if con_to_set:
      self.con_to_set = con_to_set
    else:
      contigs = self.get_contigs()
      self.con_to_set = dict(zip(contigs, len(contigs)*[0]))
      
    # set_to_con mapping
    set_to_con = {}
    for k, v in self.con_to_set.items():
      set_to_con[v] = set_to_con.get(v, []) + [k]  # invert a dictionary with non-unique values
    self.set_to_con = set_to_con
    
  def __len__(self,):
    _, L_max = self.mask_len(self.str)
    return L_max
  
  def map(self, sel, src, dst):
    '''
    Convert the contig selection in one indexing scheme to another.
    Will return None if selection is not in a contig.
    
    Input
    ----------
    sel (str): selection of a contig range or idx0 range. Can take multiple comma separated values of same type. Ex: A5-10,B2-8 or 3-8,14-21
    src (str): <'ref', 'hal'>
    dst (str): <'ref_pdb_idx', 'hal_pdb_idx', 'ref_idx0', 'hal_idx0>
    '''
    out = []
    for con in sel.split(','):
    
      ch, s, e = SampledMask.parse_contig(con)

      # selection type is pdb_idx
      if ch:
        src_long = f'{src}_pdb_idx'
        mapping = dict(zip(self.con_mappings[src_long], self.con_mappings[dst]))
        out += [mapping.get((ch, res)) for res in range(s, e+1)]

      # selection type is idx0
      else:
        src_long = f'{src}_idx0'
        mapping = dict(zip(self.con_mappings[src_long], self.con_mappings[dst]))
        out += [mapping.get(i) for i in range(s, e+1)]
      
    return out          
       
  @staticmethod
  def expand(mask_str):
    '''
    Ex: '2,A3-5,3' --> [None, None, (A,3), (A,4), (A,5), None, None, None]
    '''
    expanded = []
    for l in mask_str.split(','):
      ch, s, e = SampledMask.parse_contig(l)
      
      # contig
      if ch:
        expanded += [(ch, res) for res in range(s, e+1)]
      # gap
      else:
        expanded += [None for _ in range(s)]
    
    return expanded
  
  @staticmethod
  def contract(pdb_idx):
    '''
    Inverse of expand
    Ex: [None, None, (A,3), (A,4), (A,5), None, None, None] --> '2,A3-5,3'
    '''
    
    contracted = []
    l_prev = (None, -200)
    first_el_written = False
    
    for l_curr in pdb_idx:
      if l_curr is None:
        l_curr = (None, -100)
        
      # extend gap
      if l_curr == l_prev:
        L_gap += 1
        
      # extend con
      elif l_curr == (l_prev[0], l_prev[1]+1):
        con_e = l_curr[1]
        
      # new gap
      elif (l_curr != l_prev) and (l_curr[0] is None):
        # write prev con
        if 'con_ch' in locals():
          contracted.append(f'{con_ch}{con_s}-{con_e}')
        
        L_gap = 1
        
      # new con
      elif (l_curr != l_prev) and isinstance(l_curr[0], str):
        # write prev con
        if isinstance(l_prev[0], str) and ('con_ch' in locals()):
          contracted.append(f'{con_ch}{con_s}-{con_e}')
        # write prev gap
        elif 'L_gap' in locals():
          contracted.append(str(L_gap))

        con_ch = l_curr[0]
        con_s = l_curr[1]
        con_e = l_curr[1]
        
      # update l_prev
      l_prev = l_curr
      
    # write last element
    if isinstance(l_prev[0], str) and ('con_ch' in locals()):
      contracted.append(f'{con_ch}{con_s}-{con_e}')
    elif 'L_gap' in locals():
      contracted.append(str(L_gap))
    
    return ','.join(contracted)
    
  def subset(self, sub):
    '''
    Make a mask_str that is a subset of the original mask_str
    Ex: self.mask_str = '2,A5-20,4', sub='A5-10' --> '2,A5-10,14'
    '''
    
    # map from hal_idx0 to ref_pdb_idx
    hal_idx0 = self.map(sub, 'ref', 'hal_idx0')
    ref_pdb_idx = SampledMask.expand(sub)
    mapping = dict(zip(hal_idx0, ref_pdb_idx))
    
    expanded = [mapping.get(idx0) for idx0 in range(len(self))]      
    
    return self.contract(expanded)
  
  def mask_len(self, mask):
    '''
    Technically, can take both sampled and unsampled mask
    '''
    L_min = 0
    L_max = 0
    for l in self.str.split(','):
      ch, s, e = SampledMask.parse_contig(l)
      
      # contig
      if ch:
        L_min += e - s + 1
        L_max += e - s + 1
      # gap
      else:
        L_min += s
        L_max += e
        
    return L_min, L_max    
  
  def get_contigs(self, include_receptor=True):
    '''
    Get a list of all contigs in the mask
    '''     
    [con for con in self.str.split(',') if SampledMask.parse_contig(con)[0]]
    
    contigs = []
    for con in self.str.split(','):
      ch = SampledMask.parse_contig(con)[0]
      if ch == 'R' and include_receptor == False:
        continue
      if ch:
        contigs.append(con)
        
    return contigs
    
  def get_gaps(self,):
    '''
    Get a list of all gaps in the mask
    '''
    return [con for con in self.str.split(',') if SampledMask.parse_contig(con)[0] is None]
    
  @staticmethod
  def parse_range(_range):
    if '-' in _range:
      s, e = _range.split('-')
    else:
      s, e = _range, _range

    return int(s), int(e)

  @staticmethod
  def parse_contig(contig):
    '''
    Return the chain, start and end residue in a contig or gap str.

    Ex:
    'A4-8' --> 'A', 4, 8
    'A5'   --> 'A', 5, 5
    '4-8'  --> None, 4, 8
    'A'    --> 'A', None, None
    '''

    # is contig
    if contig[0].isalpha():
      ch = contig[0]
      if len(contig) > 1:
        s, e = SampledMask.parse_range(contig[1:])
      else:
        s, e = None, None
    # is gap
    else:
      ch = None
      s, e = SampledMask.parse_range(contig)

    return ch, s, e

  def remove_diag(self, m_2d):
    '''
    Set the diagonal of a 2D boolean array to False
    '''
    L = m_2d.shape[0]
    m_2d[np.arange(L), np.arange(L)] = False
    
    return m_2d
  
  def get_receptor_contig(self,):
    '''
    Returns None if there is no chain R in the mask_str
    '''
    receptor_contig = [l for l in self.get_contigs() if 'R' in l]
    
    if len(receptor_contig) == 0:
      receptor_contig = None
    else:
      receptor_contig = ','.join(receptor_contig)
      
    return receptor_contig
  
  def remove_receptor(self, m_2d):
    '''
    Remove intra-receptor contacts (chain R) from a mask
    '''
    receptor_contig = self.get_receptor_contig()
    
    if receptor_contig:  # has chain R
      m_1d = np.zeros(self.L_hal, dtype=bool)
      idx = np.array(self.map(receptor_contig, 'ref', 'hal_idx0'))
      m_1d[idx] = True
      update = m_1d[:, None] * m_1d[None, :]
      m_2d = m_2d * ~update 
    
    return m_2d
    
  def get_mask_con(self, include_receptor=False):
    # Make a 2D boolean mask for each contig set
    L = self.L_hal
    mask_con = np.zeros([L, L], dtype=bool)
    
    for set_id, contigs in self.set_to_con.items():
      m_1d = np.zeros(L, dtype=bool)
      for con in contigs:
        idx = self.map(con, 'ref', 'hal_idx0')
        idx = [l for l in idx if l != None]
        idx = np.array(idx, dtype=int)
        m_1d[idx] = True
      
      update = m_1d[:, None] * m_1d[None, :] 
      mask_con = np.any([mask_con, update], axis=0)
    
    # clean up
    mask_con = self.remove_diag(mask_con)
    
    if not include_receptor:
      mask_con = self.remove_receptor(mask_con)
      
    return mask_con 
  
  def get_mask_hal(self,):
    mask_hal = ~self.get_mask_con()
    mask_hal = self.remove_diag(mask_hal)
    mask_hal = self.remove_receptor(mask_hal)
    
    return mask_hal
    
  def get_mask_cce(self, pdb, cce_cutoff=20., include_receptor=False):
    '''
    Remove ij pixels where contig distances are greater than cce_cutoff.
    '''
    # start with mask_con
    mask_con = self.get_mask_con(include_receptor=include_receptor)
    
    # get ref dists
    xyz_ref = torch.tensor(pdb['xyz'][:,:3,:]).float()
    c6d_ref = geometry.xyz_to_c6d(xyz_ref[None].permute(0,2,1,3),{'DMAX':20.0}).numpy()
    dist = c6d_ref[0,:,:,0]  # (L_ref, L_ref)
    
    # scatter
    dist_scattered = self.scatter_2d(dist)
    
    # apply cce cuttoff
    update = dist_scattered < cce_cutoff
    mask_cce = np.all([mask_con, update], axis=0)

    return mask_cce
    
  def scatter_2d(self, ref_feat_2d):
    '''
    Inputs
    ---------
    ref_feat_2d (np.array; (L_ref, L_ref, ...)): Features to be scattered. The first two leading dimensions must be equal to L_ref.
    '''
    assert ref_feat_2d.shape[:2] == (self.L_ref, self.L_ref), 'ERROR: feat_2d must have leading dimensions of (L_ref, L_ref)'
    
    trailing_dims = ref_feat_2d.shape[2:]
    dtype = ref_feat_2d.dtype
    hal_feat_2d = np.zeros((self.L_hal, self.L_hal)+trailing_dims, dtype=dtype)
    
    con_hal_idx0 = np.array(self.con_mappings['hal_idx0'])
    ref_hal_idx0 = np.array(self.con_mappings['ref_idx0'])
    hal_feat_2d[con_hal_idx0[:, None], con_hal_idx0[None, :]] = ref_feat_2d[ref_hal_idx0[:, None], ref_hal_idx0[None, :]]
    
    return hal_feat_2d
  
  def scatter_1d(self, ref_feat_1d):
    '''
    Inputs
    ---------
    ref_feat_1d (np.array; (L_ref, ...)): Features to be scattered. The first leading dimension must be equal to L_ref.
    '''
    assert ref_feat_1d.shape[0] == self.L_ref, 'ERROR: feat_1d must have leading dimensions of (L_ref,)'
    
    trailing_dims = ref_feat_1d.shape[1:]
    dtype = ref_feat_1d.dtype
    hal_feat_1d = np.zeros((self.L_hal,)+trailing_dims, dtype=dtype)
    
    con_hal_idx0 = np.array(self.con_mappings['hal_idx0'])
    ref_hal_idx0 = np.array(self.con_mappings['ref_idx0'])
    hal_feat_1d[con_hal_idx0] = ref_feat_1d[ref_hal_idx0]
    
    return hal_feat_1d
  
  def idx_for_template(self, gap=200):
    '''
    Essentially return hal_idx0, except have a large jump for chain B,
    to simulate a chain break. If B contains internal jumps in residue
    numbering, these are preserved.
    '''
    
    is_rec = self.m1d_receptor()
    resi_rec = np.array([idx[1] for idx in SampledMask.expand(self.str) 
                         if idx is not None and idx[0]=='R'])
    L_binder = sum(~is_rec)


    if len(resi_rec)>0:
      if is_rec[0]:
        # receptor first
        idx_tmpl = np.arange(resi_rec[-1]+gap+1, resi_rec[-1]+gap+1+L_binder) 
        idx_tmpl = np.concatenate([resi_rec, idx_tmpl])
      else:
        # receptor second
        idx_tmpl = np.arange(L_binder)
        if resi_rec[0] <= idx_tmpl[-1]+gap:
          resi_rec += idx_tmpl[-1] - resi_rec[0] + gap + 1
        idx_tmpl = np.concatenate([idx_tmpl, resi_rec])
    else:
      #when no receptor
      idx_tmpl = np.arange(L_binder) 
    return idx_tmpl
    
  def m1d_receptor(self,):
    '''
    Get a boolean array, True if the position corresponds to the receptor
    '''
    m1d = [(l is not None) and (l[0] == 'R') for l in SampledMask.expand(self.str)]
    return np.array(m1d)
                      
  def erode(self, N_term=True, C_term=True):
    '''
    Reduce non-receptor contigs by 1 residue from the N and/or C terminus.
    '''    
    x = SampledMask.expand(self.str)
    
    if N_term:
      for i, l in enumerate(x):
        if (l is not None) and (l[0] != 'R'):
          x[i] = None
          break
          
    if C_term:
      x = x[::-1]
      
      for i, l in enumerate(x):
        if (l is not None) and (l[0] != 'R'):
          x[i] = None
          break
          
      x = x[::-1]
      
    self.str = self.contract(x)
          
    return
    
  def len_contigs(self, include_receptor=False):
    con_str = ','.join(self.get_contigs(include_receptor))
    return len(SampledMask.expand(con_str))
  
  
def make_template_features(pdb, args, device, hal_2_ref_idx0=None, sm_loss=None):
    '''
    Inputs
    ----------
    sm_loss: Instance of a contig.SampledMask object used for making the loss masks.
    '''
    PARAMS = {
        "DMIN"    : 2.0,
        "DMAX"    : 20.0,
        "DBINS"   : 36,
        "ABINS"   : 36,
    }
    if args.use_template:
        B,T = 1,1  # batch, templates

        # spoof reference features
        xyz_t = torch.tensor(pdb['xyz'][:, :3][None, None])  # (batch,templ,nres,3,3)
        t0d = torch.ones((1,1,3))  # (batch, templ, 3)

        t2d_ref = xyz_to_t2d(xyz_t=xyz_t, t0d=t0d, params=PARAMS)  # (B,T,L,L,...)
        L_ref = t2d_ref.shape[2]
        #t1d_ref = torch.ones(size=(B,T,L_ref,3), dtype=torch.float32, device=device)
        a = 2 * torch.ones([B,T,L_ref], dtype=torch.float32, device=device)
        b = 0 * torch.ones([B,T,L_ref], dtype=torch.float32, device=device)
        c = 1 * torch.ones([B,T,L_ref], dtype=torch.float32, device=device)

        t1d_ref = torch.stack([a,b,c], axis=-1)

        # Get the mask_str for scattering template features
        #1. Template mask = sampled mask
        if (args.use_template.lower() == 't') or (args.use_template.lower() == 'true'):
          sm_tmpl = sm_loss
        #2. Template mask is a subset of the sampled mask
        else:
          subset_contigs = args.use_template
          
          if args.receptor:
            receptor_contig = sm_loss.get_receptor_contig()
            subset_contigs = ','.join([subset_contigs, receptor_contig])
          
          mask_str_tmpl = sm_loss.subset(subset_contigs)            
          sm_tmpl = SampledMask(mask_str=mask_str_tmpl, ref_pdb_idx=pdb['pdb_idx'])
          
        # scatter template features
        # make leading dims (L,(L),...)
        t1d_ref = t1d_ref.permute(2,3,0,1)  # (L, ..., B, T)
        t2d_ref = t2d_ref.permute(2,3,4,0,1)  # (L, L, ..., B, T)        

        t1d_tmpl = sm_tmpl.scatter_1d(t1d_ref.cpu().numpy())
        t2d_tmpl = sm_tmpl.scatter_2d(t2d_ref.cpu().numpy())
        
        # update t2d_tmpl with mask_con (could update with mask_cce instead?)
        mask_con = sm_tmpl.get_mask_con(include_receptor=True)
        t2d_tmpl = (t2d_tmpl.T * mask_con.T).T  # trick to broadcast arrays if leading dimensions match
          
        t1d_tmpl = torch.tensor(t1d_tmpl, device=device)
        t2d_tmpl = torch.tensor(t2d_tmpl, device=device)
        
        # Permute B and T dims back to front
        t1d_tmpl = t1d_tmpl.permute(2,3,0,1)
        t2d_tmpl = t2d_tmpl.permute(3,4,0,1,2)
        
        # Make last 3 idx of last dim all 1 to mimick Ivan's template feature
        t2d_tmpl[..., -3:] = 1.

        idx = torch.tensor(sm_tmpl.idx_for_template(gap=200), device=device)[None]
        
        net_kwargs = {
            'idx': idx,
            't1d': t1d_tmpl,
            't2d': t2d_tmpl
        }

    elif args.template_pdbs is not None:
        B,T = 1, len(args.template_pdbs)  # batch, templates

        # get xyz features of all templates
        xyz_t = [torch.tensor(parse_pdb(f_pdb)['xyz'][:, :3]) for f_pdb in args.template_pdbs]
        xyz_t = torch.stack(xyz_t, axis=0)[None]  # (batch, template, nres, 3, 3)
        t0d = torch.ones(B,T,3)

        t2d_tmpl = xyz_to_t2d(xyz_t=xyz_t, t0d=t0d, params=PARAMS).to(device)  # (B,T,L,L,...)
        L_tmpl = t2d_tmpl.shape[2]
        t1d_tmpl = torch.ones(size=(B,T,L_tmpl,3), dtype=torch.float32, device=device)

        # spoof pdb idx
        idx_tmpl = torch.range(0, L_tmpl-1, dtype=torch.long, device=device)[None]

        # Net() kwargs
        net_kwargs = {
            'idx': idx_tmpl,
            't1d': t1d_tmpl,
            't2d': t2d_tmpl
        }

    else:
        net_kwargs = {}

    return net_kwargs
