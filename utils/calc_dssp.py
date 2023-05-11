#@title get secondary structure (SSE) from given PDB file
#@markdown So far it seems the best solution is to steal code from biotite
#@markdown which calculates the SSE of a peptide chain based on the P-SEA algorithm (Labesse 1997)
# CODE FROM BIOKITE
# From Krypton
import numpy as np
import random
import torch

def vector_dot(v1,v2):
    return (v1*v2).sum(axis=-1)

def norm_vector(v):
    factor = np.linalg.norm(v, axis=-1)
    if isinstance(factor, np.ndarray):
        v /= factor[..., np.newaxis]
    else:
        v /= factor
    return v

def coord(x):
  return np.asarray(x)
def displacement(atoms1, atoms2):
    v1 = coord(atoms1)
    v2 = coord(atoms2)
    if len(v1.shape) <= len(v2.shape):
        diff = v2 - v1
    else:
        diff = -(v1 - v2)
    return diff
def distance(atoms1, atoms2):
    diff = displacement(atoms1, atoms2)
    return np.sqrt(vector_dot(diff, diff))

def angle(atoms1, atoms2, atoms3):
    v1 = displacement(atoms1, atoms2)
    v2 = displacement(atoms3, atoms2)
    norm_vector(v1)
    norm_vector(v2)
    return np.arccos(vector_dot(v1,v2))

def dihedral(atoms1, atoms2, atoms3, atoms4):
    v1 = displacement(atoms1, atoms2)
    v2 = displacement(atoms2, atoms3)
    v3 = displacement(atoms3, atoms4)
    norm_vector(v1)
    norm_vector(v2)
    norm_vector(v3)
    
    n1 = np.cross(v1, v2)
    n2 = np.cross(v2, v3)
    
    # Calculation using atan2, to ensure the correct sign of the angle 
    x = vector_dot(n1,n2)
    y = vector_dot(np.cross(n1,n2), v2)
    return np.arctan2(y,x)

def replace_letters(arr):
  # Create a dictionary that maps the letters 'a', 'b', and 'c' to the corresponding numbers
  letter_to_number = {'a': 0, 'b': 1, 'c': 2}
  
  # Create a new array that will hold the numbers
  nums = []
  
  # Loop through the input array and replace the letters with the corresponding numbers
  for letter in arr:
    if letter in letter_to_number:
      nums.append(letter_to_number[letter])
    else:
      nums.append(letter)
  
  return np.array(nums)

def replace_with_mask(arr, percentage, replace_loops=False):
  # Make sure the percentage is between 0 and 100
  percentage = min(max(percentage, 0), 100)
  
  # Calculate the number of values to replace
  num_to_replace = int(len(arr) * percentage / 100)
  
  # Choose a random subset of the array to replace
  replace_indices = random.sample(range(len(arr)), num_to_replace)
  
  # Replace the values at the chosen indices with the number 3
  for i in replace_indices:
    arr[i] = 3

  if replace_loops:
    for i in arr:
        if arr[i] == 2:
            arr[i] = 3
  
  return arr

def annotate_sse(ca_coord, percentage_mask=0, replace_loops=False):
  _radians_to_angle = 2*np.pi/360

  _r_helix = ((89-12)*_radians_to_angle, (89+12)*_radians_to_angle)
  _a_helix = ((50-20)*_radians_to_angle, (50+20)*_radians_to_angle)
  _d2_helix = ((5.5-0.5), (5.5+0.5))
  _d3_helix = ((5.3-0.5), (5.3+0.5))
  _d4_helix = ((6.4-0.6), (6.4+0.6))

  _r_strand = ((124-14)*_radians_to_angle, (124+14)*_radians_to_angle)
  _a_strand = ((-180)*_radians_to_angle, (-125)*_radians_to_angle,
              (145)*_radians_to_angle, (180)*_radians_to_angle)
  _d2_strand = ((6.7-0.6), (6.7+0.6))
  _d3_strand = ((9.9-0.9), (9.9+0.9))
  _d4_strand = ((12.4-1.1), (12.4+1.1))

  # Filter all CA atoms in the relevant chain.

  d2i_coord = np.full(( len(ca_coord), 2, 3 ), np.nan)
  d3i_coord = np.full(( len(ca_coord), 2, 3 ), np.nan)
  d4i_coord = np.full(( len(ca_coord), 2, 3 ), np.nan)
  ri_coord = np.full(( len(ca_coord), 3, 3 ), np.nan)
  ai_coord = np.full(( len(ca_coord), 4, 3 ), np.nan)
  
  # The distances and angles are not defined for the entire interval,
  # therefore the indices do not have the full range
  # Values that are not defined are NaN
  for i in range(1, len(ca_coord)-1):
      d2i_coord[i] = (ca_coord[i-1], ca_coord[i+1])
  for i in range(1, len(ca_coord)-2):
      d3i_coord[i] = (ca_coord[i-1], ca_coord[i+2])
  for i in range(1, len(ca_coord)-3):
      d4i_coord[i] = (ca_coord[i-1], ca_coord[i+3])
  for i in range(1, len(ca_coord)-1):
      ri_coord[i] = (ca_coord[i-1], ca_coord[i], ca_coord[i+1])
  for i in range(1, len(ca_coord)-2):
      ai_coord[i] = (ca_coord[i-1], ca_coord[i],
                      ca_coord[i+1], ca_coord[i+2])
  
  d2i = distance(d2i_coord[:,0], d2i_coord[:,1])
  d3i = distance(d3i_coord[:,0], d3i_coord[:,1])
  d4i = distance(d4i_coord[:,0], d4i_coord[:,1])
  ri = angle(ri_coord[:,0], ri_coord[:,1], ri_coord[:,2])
  ai = dihedral(ai_coord[:,0], ai_coord[:,1],
                ai_coord[:,2], ai_coord[:,3])
  
  sse = np.full(len(ca_coord), "c", dtype="U1")
  
  # Annotate helices
  # Find CA that meet criteria for potential helices
  is_pot_helix = np.zeros(len(sse), dtype=bool)
  for i in range(len(sse)):
      if (
              d3i[i] >= _d3_helix[0] and d3i[i] <= _d3_helix[1]
          and d4i[i] >= _d4_helix[0] and d4i[i] <= _d4_helix[1]
          ) or (
              ri[i] >= _r_helix[0] and ri[i] <= _r_helix[1]
          and ai[i] >= _a_helix[0] and ai[i] <= _a_helix[1]
          ):
              is_pot_helix[i] = True
  # Real helices are 5 consecutive helix elements
  is_helix = np.zeros(len(sse), dtype=bool)
  counter = 0
  for i in range(len(sse)):
      if is_pot_helix[i]:
          counter += 1
      else:
          if counter >= 5:
              is_helix[i-counter : i] = True
          counter = 0
  # Extend the helices by one at each end if CA meets extension criteria
  i = 0
  while i < len(sse):
      if is_helix[i]:
          sse[i] = "a"
          if (
              d3i[i-1] >= _d3_helix[0] and d3i[i-1] <= _d3_helix[1]
              ) or (
              ri[i-1] >= _r_helix[0] and ri[i-1] <= _r_helix[1]
              ):
                  sse[i-1] = "a"
          sse[i] = "a"
          if (
              d3i[i+1] >= _d3_helix[0] and d3i[i+1] <= _d3_helix[1]
              ) or (
              ri[i+1] >= _r_helix[0] and ri[i+1] <= _r_helix[1]
              ):
                  sse[i+1] = "a"
      i += 1
  
  # Annotate sheets
  # Find CA that meet criteria for potential strands
  is_pot_strand = np.zeros(len(sse), dtype=bool)
  for i in range(len(sse)):
      if (    d2i[i] >= _d2_strand[0] and d2i[i] <= _d2_strand[1]
          and d3i[i] >= _d3_strand[0] and d3i[i] <= _d3_strand[1]
          and d4i[i] >= _d4_strand[0] and d4i[i] <= _d4_strand[1]
          ) or (
              ri[i] >= _r_strand[0] and ri[i] <= _r_strand[1]
          and (   (ai[i] >= _a_strand[0] and ai[i] <= _a_strand[1])
                or (ai[i] >= _a_strand[2] and ai[i] <= _a_strand[3]))
          ):
              is_pot_strand[i] = True
  # Real strands are 5 consecutive strand elements,
  # or shorter fragments of at least 3 consecutive strand residues,
  # if they are in hydrogen bond proximity to 5 other residues
  pot_strand_coord = ca_coord[is_pot_strand]
  is_strand = np.zeros(len(sse), dtype=bool)
  counter = 0
  contacts = 0
  for i in range(len(sse)):
      if is_pot_strand[i]:
          counter += 1
          coord = ca_coord[i]
          for strand_coord in ca_coord:
              dist = distance(coord, strand_coord)
              if dist >= 4.2 and dist <= 5.2:
                  contacts += 1
      else:
          if counter >= 4:
              is_strand[i-counter : i] = True
          elif counter == 3 and contacts >= 5:
              is_strand[i-counter : i] = True
          counter = 0
          contacts = 0
  # Extend the strands by one at each end if CA meets extension criteria
  i = 0
  while i < len(sse):
      if is_strand[i]:
          sse[i] = "b"
          if d3i[i-1] >= _d3_strand[0] and d3i[i-1] <= _d3_strand[1]:
              sse[i-1] = "b"
          sse[i] = "b"
          if d3i[i+1] >= _d3_strand[0] and d3i[i+1] <= _d3_strand[1]:
              sse[i+1] = "b"
      i += 1
  sse=replace_letters(sse)
  sse=replace_with_mask(sse, percentage_mask, replace_loops=replace_loops)
  sse=torch.nn.functional.one_hot(torch.tensor(sse), num_classes=4)
  return sse
