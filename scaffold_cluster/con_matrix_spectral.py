import numpy as np
from rdkit import Chem
import h5py
from sklearn.cluster import SpectralClustering
from collections import Counter

smiles_list = []
with open("./scaffold.smi", 'r') as f:
    for i, line in enumerate(f):
        smiles = line
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            smiles_list.append(Chem.MolToSmiles(mol))
            #if i >1000:
            #    break

smiles_list = list(set(smiles_list))

#numpy 在cpu上拼接矩阵
#with h5py.File('h5file_com4_11_km.h5', 'r') as hf:
#    data11 = np.array(hf['elem'])

#with h5py.File('h5file_com4_12_km.h5', 'r') as hf:
#    data12 = np.array(hf['elem'])

#with h5py.File('h5file_com4_21_km.h5', 'r') as hf:
#    data21 = np.array(hf['elem'])

#with h5py.File('h5file_com4_22_km.h5', 'r') as hf:
#    data22 = np.array(hf['elem'])

#tani_index1 = np.concatenate((data11,data12),1)
#tani_index2 = np.concatenate((data21, data22), 1)
#tani_matrix = np.concatenate((tani_index1,tani_index2),0)

#with h5py.File('h5file_com4_all_km.h5', 'w') as hf:
#    hf.create_dataset('elem', data=tani_matrix, compression='gzip', compression_opts=4)

#读拼接后的矩阵进行谱聚类
with h5py.File('h5file_com4_all_km.h5', 'r') as hf:
     data  = np.array(hf['elem'])
print("read end and begin cluster")
sc = SpectralClustering(30, affinity='precomputed', n_init=10, assign_labels='discretize')
sc.fit(data)
print("cluster end")
print(Counter(sc.labels_))
label = sc.labels_
label =label.tolist()
label.pop(47307)
#print(sc_labels_)
with open("./scaffold_cluster_1.0.smi", "w") as f:
    for i in range(len(smiles_list)):
       f.write(smiles_list[i] + " " + str(label[i]) + '\n')
