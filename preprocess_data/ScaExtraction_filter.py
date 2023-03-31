import numpy as np
import scaffoldgraph as sg
import networkx as nx
import os
from rdkit.Chem import Draw
from rdkit import Chem
from utils import get_mol
import matplotlib.pyplot as plt

def filter_sca(sca):
    mol = Chem.MolFromSmiles(sca)
    ri = mol.GetRingInfo()
    if ri.NumRings() <2:
        return None
    elif mol.GetNumHeavyAtoms() >20:
        return None
    elif Chem.rdMolDescriptors.CalcNumRotatableBonds(mol) >3:
        return None
    else:
        return True

#loss.forward bug:mol equal sca,index tensor of side is null ,default float32 not int so error add_index need int but get float
def mol_if_equal_sca (mol,sca):
    S_sca = []
    smile = get_mol(mol)
    sca = get_mol(sca)
    if mol == None or sca == None :
        return False
    else:
        n_atoms = smile.GetNumAtoms()
        index = smile.GetSubstructMatch(sca)
        for i in range(n_atoms):
            if i in index:
                S_sca.append(1)
            else:
                S_sca.append(0)
        arr = np.array(S_sca)
        if (arr == 1).all() == True or (arr == 0).all() == True:
            return False
        else:
            return True

if __name__ == "__main__":

    smi_file = "D:\Python\ProjectOne\data\\all_ZINC.smi"
    network = sg.ScaffoldNetwork.from_smiles_file(smi_file)
    n_scaffolds = network.num_scaffold_nodes
    n_molecules = network.num_molecule_nodes
    print('\nGenerated scaffold network from {} molecules with {} scaffolds\n'.format(n_molecules, n_scaffolds))
    scaffolds = list(network.get_scaffold_nodes())
    molecules = list(network.get_molecule_nodes())
    filter_sca_num = 0

    counts = network.get_hierarchy_sizes()  # returns a collections Counter object
    lists = sorted(counts.items())
    x, y = zip(*lists)
    plt.figure(figsize=(8, 6))
    plt.bar(x, y)
    plt.xlabel('Hierarchy')
    plt.ylabel('Scaffold Count')
    plt.title('Number of Scaffolds per Hierarchy (Network)')
    plt.show()

    with open ("data\scaffold.smi" , "w") as f:
        for i in scaffolds:
            if filter_sca(i) is not None:
                f.write(i + '\n')
                filter_sca_num+=1
    print('\nfilter scaffold with {} scaffolds\n'.format(filter_sca_num))
    with open ("data\mol_sca.smi" , "w") as f:
        total = 0
        for pubchem_id in molecules:
            predecessors = list(nx.bfs_tree(network, pubchem_id, reverse=True))
            smile = network.nodes[predecessors[0]]['smiles']
            sca =0
            for i in range(1,len(predecessors)):
                if filter_sca(predecessors[i]) is not None:
                    sca = predecessors[i]
                    break
            if sca !=0 and mol_if_equal_sca(smile,sca):
                f.write(smile + " " + sca +'\n')
                total +=1

    print('\nthe pairs of molecula and scaffold have {} \n'.format(total))

