import scipy.sparse
import numpy as np
import pandas as pd
import scipy.sparse
from rdkit import Chem ,DataStructs
from rdkit.Chem import MACCSkeys
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect as Morgan
from multiprocessing import Pool
import torch
import time
import scipy
import h5py
from rdkit.Chem import AllChem

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

if __name__ == "__main__":

    smiles_list = ["O=C(CNC(=O)c1ccco1)OCC(=O)c1ccc2ccccc2c1"]

    with open("D:\Python\ProjectOne2.0\data\sample_kl_0.1_4.csv", 'r') as f:
        for i, line in enumerate(f):
            smiles = line
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                smiles_list.append(Chem.MolToSmiles(mol))

    print(len(smiles_list))
    maccs_fps = [AllChem.GetMACCSKeysFingerprint(get_mol(mol)) for mol in smiles_list]
    maccs = DataStructs.BulkTanimotoSimilarity(maccs_fps[0], maccs_fps[1:])

    data = pd.read_csv('D:\Python\ProjectOne2.0\data\sample_kl_0.1_4.csv')
    data["similarty"] = maccs
    data.to_csv('D:\Python\ProjectOne2.0\data\sample_similarty_kl_0.1_4.csv', index=False)
