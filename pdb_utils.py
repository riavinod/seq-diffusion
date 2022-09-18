from prody import *
from pylab import *
import os
from Bio import SeqIO
import re
import Bio.PDB.PDBParser
import csv
import pandas as pd
import json
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torchvision
import torch



def get_pdb_ids(id):
    '''For a given CATH ID, return the list of PDB IDs; append these IDs to PDB List.'''
    
    node = cath.find(id)
    pdb_id_list = node.getPDBs()
    
    print('CATH ID: ' + id + ', PDBs: ' , len(pdb_id_list))
    
    pdb_id_list = [p[:4] for p in pdb_id_list]
    
    return set(pdb_id_list)


def get_pdb_file(pdb_id):
    '''Given a PDB ID (name) return the PDB object.'''
    ag = parsePDB(pdb_id)

    os.system('mv *.gz pdb_files/')
    os.system('gunzip pdb_files/*.gz')

    return ag

def get_atomic(pdb_id, ag):
    '''Given a PDB ID (name) return the XYZ coordinates of the backbone (includes C-alpha atoms).'''
    
    coords = ag.backbone.getCoords()
    num_bb_atoms = ag.backbone.numAtoms()

    return coords, num_bb_atoms


def get_sequence(ag):
    seq = list(ag.getSequence())
    idx = list(ag.backbone.getIndices())

    seq_filtered = list(array(seq)[idx])
    seq_filtered = seq_filtered[0:len(seq_filtered):4]

    return seq_filtered, ag.calpha.numAtoms()

def validate_pdb(seq, ag):
    n = ag.calpha.numAtoms()
    return (len(seq) != n)

def get_sequence_test(pdb_id):
    '''Given a PDB ID (name) return the primary sequence; return seq and its length.'''

    d3to1 = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
    'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N', 
    'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W', 
    'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}

    #record = 'pdb_files/'+pdb_id+'.pdb'
    record = pdb_id.lower()+'.pdb'

    # run parser
    parser = Bio.PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('struct', record)    

    # iterate each model, chain, and residue
    # printing out the sequence for each chain
    count = 0

    for model in structure:
        for chain in model:
            seq = []
            for residue in chain:
                print(residue.resname)
                count+=1
                seq.append(d3to1[residue.resname])
            #print('>some_header\n',''.join(seq))
    return seq, count

def array_to_dir(data, pdb_id, S, arc_class):
    '''Write array to a file.'''

    np.save("data/"+S+"/"+arc_class+'/'+pdb_id, data)

# https://github.com/aqlaboratory/openfold/blob/e8b3789f3320aac6be11833cc912e659a53e7874/run_pretrained_openfold.py#L381
def parse_fasta(pdb_id, arc_class):

    with open(os.path.join("data/fasta/"+arc_class+'/', pdb_id+".fasta"), "r") as fp:
        data = fp.read()

    data = re.sub('>$', '', data, flags=re.M)
    lines = [
        l.replace('\n', '')
        for prot in data.split('>') for l in prot.strip().split('\n', 1)
    ][1:]
    tags, seqs = lines[::2], lines[1::2]

    tags = [t.split()[0] for t in tags]

    return tags, seqs


def structure_lookup(pdb_id):
    with open('pdb_meta/pdb_label_map.json') as json_file:
        dict = json.load(json_file)
    
    return dict[pdb_id]


def get_pdb_list():
    with open("pdb_meta/pdb_ids.txt") as file_in:
        lines = []
        for line in file_in:
            lines.append(line[:-1])
    return lines

    # with open("pdb_meta/pdb_ids.txt","r") as f:
    #     rd = csv.reader(f)
    #     pdb_list = list(rd)[0]
    # return pdb_list

def get_pdb_label_map(n):
    pdbs = open("pdb_meta/pdb_ids.txt", "r")
    pdbs = pdbs.read().split("\n")

    labels = open("pdb_meta/pdb_labels.txt", "r")
    labels = labels.read().split("\n")

    map = dict(zip(pdbs, labels))

    class_counts = {}
    
    for s in set(labels):
        count = labels.count(s)
        class_counts[s] = count
    
    sorted_class_counts = sorted(class_counts, key=class_counts.get, reverse=True)
    top_n = []
    for i in range(n):
        top_n.append(sorted_class_counts[i])

    return  top_n


def makedir(path):
    try:
        os.makedirs(path)
    except OSError:
        pass



def class_map(y, structs, dir):
    
    values = range(len(set(y))) # y could be structures or numbers
    if dir: # map structure name to number
        map = dict(zip(list(y), values)) 
    else: # map number to structure name
         map = dict(zip(values, structs))
    #print(map)
    
    y_vec = []
    for label in y:
        y_vec.append(map[label]) 
        #print(map[label])
    y_vec = np.array(y_vec)
    return y_vec

def name2number(y):
    numbers = range(len(set(y)))
    names = set(y)
    map = dict(zip(names, numbers)) 
    
    mapped_y = np.array([map[struct] for struct in y])

    return map, mapped_y



class PDBDataset(Dataset):
    
    def __init__(self, file_path_x, file_path_h, file_path_y, transform=None):
        self.positions = np.load(file_path_x)
        self.nodes = np.load(file_path_h)
        
        y_structs = np.load(file_path_y)
        # structs = set(list(np.load(file_path_y)))
        
        self.map, self.labels = name2number(y_structs)
        self.transform = transform

        # print('positions', self.positions.shape)
        # print('nodes', self.nodes.shape)
        # print('labels', self.labels.shape)
        
    def __len__(self):
        return len(self.positions)
    
    def __getitem__(self, index):
        # one image == one residue
        # load image as ndarray type (Height * Width * Channels)
        # be carefull for converting dtype to np.uint8 [Unsigned integer (0 to 255)]
        # in this example, i don't use ToTensor() method of torchvision.transforms
        # so you can convert numpy ndarray shape to tensor in PyTorch (H, W, C) --> (C, H, W)
        #residue = self.data[index].astype(np.uint8).reshape((4993, 4, 4)) # this is for numpy array
        position = self.positions[index].astype(np.uint8).reshape((128, 3)) # this is for tensor
        node = self.nodes[index].astype(np.uint8).reshape((128, 256))
        label = self.labels[index]
        
        if self.transform is not None:
            position = self.transform(position)
            node = self.transform(node)
        
        sample = {'position': position, 'node': node, 'label': label}

        return sample


def get_dataloaders():

    train_dataset = PDBDataset('data/train/X0.npy', 'data/train/H0.npy', 'data/train/y0.npy', transform=torchvision.transforms.ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    test_dataset = PDBDataset('data/test/X0.npy', 'data/test/H0.npy', 'data/test/y0.npy', transform=torchvision.transforms.ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    valid_dataset = PDBDataset('data/valid/X0.npy', 'data/valid/H0.npy', 'data/valid/y0.npy', transform=torchvision.transforms.ToTensor())
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=True)

    dataloaders = { 'train' : train_loader,
                    'test' : test_loader,     
                    'valid' : valid_loader,
                 }


    return dataloaders


if __name__ == "__main__":

    dataloader = get_dataloaders()['train']
    
    for i, data in enumerate(dataloader):
        print(data.keys())
        print(data['position'].shape)
        print(data['node'].shape)
        print(data['label'].shape)

        break


def compute_mean_mad(dataloaders, label_property):
    values = torch.tensor(dataloaders['train'].dataset.labels).float()

    
    
    #values = dataloaders['train'].data[label_property]
    meann = torch.mean(values)
    ma = torch.abs(values - meann)
    mad = torch.mean(ma)
    return meann, mad