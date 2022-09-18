from prody import *
from pylab import *
import os
from Bio import SeqIO
import re
import Bio.PDB.PDBParser
import csv
import json
import pdb_utils

'''Convert PDB meta data to PDB data {name, sequence, ca_coords}'''

pdb_list = pdb_utils.get_pdb_list()
print('Number of PDBs:', len(pdb_list))

print(pdb_list)

pdb_data = []

for i in range(len(pdb_list)):
    ag = pdb_utils.get_pdb_file(pdb_list[i])
    seq, num_ca = pdb_utils.get_sequence(ag)
    seq = ''.join(seq)
    ca_coords = ag.calpha.getCoords()/10
    
    dict = {}

    dict['name'] = pdb_list[i]
    dict['sequence'] = seq
    dict['ca_coords'] = ca_coords.tolist()

    pdb_data.append(dict)

    print(pdb_list[i])
    print(len(seq))
    print(ca_coords.shape)


with open("data/pdb_data.json", "w") as outfile:
    json.dump(pdb_data, outfile)