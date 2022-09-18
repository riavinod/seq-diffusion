from prody import *
from pylab import *
import os
from Bio import SeqIO
import re
import Bio.PDB.PDBParser
import csv
import pandas as pd
import json

#Requires (1) pdb_names.txt        -- from  https://www.rcsb.org/
#         (2) cath-domain-list.txt -- from http://www.cathdb.info ‘latest release statistics’  '''

cath_dict = {

# mainly alpha

'1.10' : 'orthogonal bundle',
'1.20' : 'up-down bundle',
'1.25' : 'alpha horseshoe',
'1.40' : 'alpha solenoid',
'1.50' : 'alpha/alpha barrel',

# mainly beta

'2.10' : 'ribbon',
'2.20' : 'single sheet',
'2.30' : 'roll',
'2.40' : 'beta barrel',
'2.50' : 'clam',
'2.60' : 'sandwich',
'2.70' : 'distorted sandwich',
'2.80' : 'trefoil',
'2.90' : 'orthogonal prism',
'2.100' : 'aligned prism',
'2.102' : '3-layer sandwich',
'2.105' : '3 propeller',
'2.110' : '4 propeller',
'2.115' : '5 propeller',
'2.120' : '6 propeller',
'2.130' : '7 propeller',
'2.140' : '8 propeller',
'2.150' : '2 solenoid',
'2.160' : '3 solenoid',
'2.170' : 'beta complex',
'2.180' : 'shell',

# alpha beta

'3.10' : 'roll',
'3.15' : 'super roll',
'3.20' : 'alpha-beta barrel',
'3.30' : '2-layer sandwich',
'3.40' : '3-layer(aba) sandwich',
'3.50' : '3-layer(bba) sandwich',
'2.55' : '3-layer(bab) sandwich' ,
'3.60' : '4-layer sandwich',
'3.65' : 'alpha-beta prism',
'3.70' : 'box',
'3.75' : '5-stranded propeller',
'3.80' : 'alpha-beta horseshoe',
'3.90' : 'alpha-beta complex',
'3.100' : 'ribosomal protein l15; chain:k; domain 2'   
}



if __name__ == "__main__":

    # structs = get_pdb_label_map(7)
    # print(structs)


    # get list of PDB ID names
    with open("pdb_meta/prot_diff_pdbs.txt","r") as f:
        rd = csv.reader(f)
        pdb_list = list(rd)[0]

    txt_file = r'pdb_meta/cath-domain-list.txt'
    tsv_file = r'pdb_meta/cath-domain-list.tsv'


    with open(txt_file, 'r') as infile, open(tsv_file, 'w') as outfile:
        stripped = (line.strip() for line in infile)
        lines = (line.split(",") for line in stripped if line)
        writer = csv.writer(outfile)
        writer.writerows(lines)

    cath_domains = pd.read_csv('pdb_meta/cath-domain-list.tsv', sep="\s{2,}", engine = 'python')
    cath_domains['structure_id'] = cath_domains['C'].astype(str) + '.' + cath_domains['A'].astype(str)

    missing_pdbs = []
    other_structs = []
    struct_labels = []
    pdb_label_map = {}

    # get CATH architecture classification for each PDB
    for p in pdb_list:
        pdb_domain = p.lower() +'A00'
        if pdb_domain not in list(cath_domains['pdb_domain']):
            missing_pdbs.append(p)
        else:
            item = cath_domains.loc[cath_domains['pdb_domain'] == pdb_domain, 'structure_id'].item()
            if item not in cath_dict.keys():
                other_structs.append(p)
            else:    
                print('PDB: ' + str(p) + ' ----> ' + cath_dict[item])
                struct_labels.append(cath_dict[item])
                pdb_label_map[p] = cath_dict[item]

    print(str(len(missing_pdbs)) + '/ ' + str(len(pdb_list)))
    print('PDBs not in CATH file: ' + str(len(missing_pdbs)))
    print('PDBs with special structures: ' + str(len(other_structs)))

    with open('pdb_meta/pdb_label_map.json', 'w') as convert_file:
        convert_file.write(json.dumps(pdb_label_map))

    # write a new list of PDB ID names based on label availability
    filter_by_label_pdbs = list(set(pdb_list) - set(missing_pdbs) - set(other_structs))

    struct_types = set(struct_labels)
    for s in struct_types:
        count = struct_labels.count(s)
        print(s + ': ' + str(count))

    with open(r'pdb_meta/pdb_ids.txt', 'w') as fp:
        for item in filter_by_label_pdbs:
            # write each item on a new line
            fp.write("%s\n" % item)

    with open(r'pdb_meta/pdb_labels.txt', 'w') as fp:
        for item in struct_labels:
            # write each item on a new line
            fp.write("%s\n" % item)