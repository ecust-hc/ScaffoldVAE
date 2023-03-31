from rdkit import Chem ,DataStructs

if __name__ == "__main__":

    ref = ["O=C(CNC(=O)c1ccco1)OCC(=O)c1ccc2ccccc2c1"]
    sca_list = ["c1ccc2ccccc2c1"]
    smiles_list =[]
    sim = []
    with open("D:\Python\ProjectOne2.0\data\sample_similarty_kl_0.1_4.csv", 'r') as f:
        for i, line in enumerate(f):
            smiles = line.split(",")[0]
            smilarty = line.split(",")[1]
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                smiles_list.append(Chem.MolToSmiles(mol))
                sim.append(smilarty)
    m = Chem.MolFromSmiles(ref[0])

    patt = Chem.MolFromSmarts(sca_list[0])
    rm = Chem.DeleteSubstructs(m, patt)
    frag = Chem.MolToSmiles(rm)
    num =0
    filter = []
    filter_sim =[]
    print(frag)
    for i in range(len(smiles_list)):
        mol = Chem.MolFromSmiles(smiles_list[i])
        if mol.HasSubstructMatch(rm):
            num +=1
            filter.append(Chem.MolToSmiles(mol))
            filter_sim.append(sim[i])
    with open("data\sample_filter_kl_0.1_4.smi", "w") as f:
        for i in range(len(filter)):
            f.write(filter[i] +" "+ filter_sim[i] )
    print("filter {}".format(num/len(smiles_list)))
