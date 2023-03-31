
with open("./data/mol_sca.smi", 'r') as f:
    smiles_list = []
    scas_list = []
    for i, line in enumerate(f):
        smiles = line.split(" ")[0]
        scas = line.split(" ")[1].rstrip("\n")
        smiles_list.append(smiles)
        scas_list.append(scas)

with open("D:\Python\ProjectOne\data\scaffold_cluster_1.0.smi", 'r') as f:
    scas = []
    cluster = []
    for i, line in enumerate(f):
        sca = line.split(" ")[0]
        id = line.split(" ")[1].rstrip("\n")
        scas.append(sca)
        cluster.append(id)

mol_index = []
for i in range(len(scas_list)):
    if scas_list[i] in scas:
        index = scas.index(scas_list[i])
        mol_index.append(cluster[index])

with open ("data\mol_sca_cluster.smi" , "w") as f:
    for i in range(len(smiles_list)):
        f.write(smiles_list[i] +  " " + scas_list[i] + " " +str(mol_index[i]) + '\n')

print("Done")