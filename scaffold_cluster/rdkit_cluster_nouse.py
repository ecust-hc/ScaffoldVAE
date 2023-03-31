from rdkit import rdBase, Chem, DataStructs
from rdkit.Chem import AllChem

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
smiles_list = []
def get_mol(smiles_or_mol):
    '''
    Loads SMILES/molecule into RDKit's object
    '''
    if isinstance(smiles_or_mol, str):
        if len(smiles_or_mol) == 0:
            return None
        mol = Chem.MolFromSmiles(smiles_or_mol)
        if mol is None:
            return None
        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            return None
        return mol
    return smiles_or_mol
with open("D:\Python\ProjectOne\data\scaffold.smi", 'r') as f:
    for i, line in enumerate(f):
        smiles = line
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            smiles_list.append(Chem.MolToSmiles(mol))
smiles_list = list(set(smiles_list))
print(len(smiles_list))


morgan_fp = [AllChem.GetMorganFingerprintAsBitVect(get_mol(x),2,2048) for x in smiles_list]
dis_matrix = [DataStructs.BulkTanimotoSimilarity(morgan_fp[i], morgan_fp[:20000],returnDistance=True) for i in range(20000)]
dis_array = np.array(dis_matrix)
ward = AgglomerativeClustering(n_clusters=20)
ward.fit(dis_array)
print(ward.labels_.shape)