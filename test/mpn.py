import torch
import torch.nn as nn
import rdkit.Chem as Chem
import torch.nn.functional as F
from torch.autograd import Variable
from rdkit.Chem.Scaffolds import MurckoScaffold
import re
from model import DMPN

SMILES_BATCH = ["CCC(NC(=O)c1scnc1C1CC1)C(=O)N1CCOCC1","CCN(C)S(=O)(=O)N1CCC(Nc2cccc(OC)c2)CC1","Cc1ccc(SCC(=O)OCC(=O)NC2(C#N)CCCCC2)cc1","N#Cc1nc(-c2ccccc2F)oc1NCc1ccccc1"]
#SMILES_BATCH = ["CCC(NC(=O)c1scnc1C1CC1)C(=O)N1CCOCC1"]
ELEM_LIST = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'Al', 'I', 'B', 'K', 'Se', 'Zn',
             'H', 'Cu', 'Mn', 'unknown']
ATOM_FDIM = len(ELEM_LIST) + 6 + 5 + 4 + 1
BOND_FDIM = 5 + 6
MAX_NB = 6

def get_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    Chem.Kekulize(mol)
    return mol

def create_var(tensor, requires_grad=None):
    if requires_grad is None:
        return Variable(tensor).cuda()
    else:
        return Variable(tensor, requires_grad=requires_grad).cuda()

def index_select_ND(source, dim, index):
    index_size = index.size()
    suffix_dim = source.size()[1:]
    final_size = index_size + suffix_dim
    target = source.index_select(dim, index.view(-1))
    return target.view(final_size)

def onek_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def atom_features(atom):
    return torch.Tensor(onek_encoding_unk(atom.GetSymbol(), ELEM_LIST)
                        + onek_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5])
                        + onek_encoding_unk(atom.GetFormalCharge(), [-1, -2, 1, 2, 0])
                        + onek_encoding_unk(int(atom.GetChiralTag()), [0, 1, 2, 3])
                        + [atom.GetIsAromatic()])

def bond_features(bond):
    bt = bond.GetBondType()
    stereo = int(bond.GetStereo())
    fbond = [bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE, bt == Chem.rdchem.BondType.TRIPLE,
             bt == Chem.rdchem.BondType.AROMATIC, bond.IsInRing()]
    fstereo = onek_encoding_unk(stereo, [0, 1, 2, 3, 4, 5])
    return torch.Tensor(fbond + fstereo)

def atom_if_sca(mol_batch,sca_batch):
    S_sca = []
    scope = []
    total_atoms = 0
    for i in range(len(mol_batch)):
        smile = mol_batch[i]
        smile = get_mol(smile)
        sca = sca_batch[i]
        sca = get_mol(sca)
        n_atoms = smile.GetNumAtoms()
        index = smile.GetSubstructMatch(sca)
        for i in range(n_atoms):
            if i in index:
                S_sca.append(1)
            else:
                S_sca.append(0)
        scope.append((total_atoms, n_atoms))
        total_atoms += n_atoms
    return S_sca


def mol2graph(mol_batch):
    padding = torch.zeros(BOND_FDIM)
    fatoms, fbonds = [], [padding]  # Ensure bond is 1-indexed
    out_bonds,in_bonds, all_bonds = [], [], [(-1, -1)]  # Ensure bond is 1-indexed
    scope = []
    total_atoms = 0

    for smiles in mol_batch:
        mol = get_mol(smiles)
        # mol = Chem.MolFromSmiles(smiles)
        n_atoms = mol.GetNumAtoms()
        for atom in mol.GetAtoms():
            fatoms.append(atom_features(atom))
            in_bonds.append([])
            out_bonds.append([])

        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()
            x = a1.GetIdx() + total_atoms
            y = a2.GetIdx() + total_atoms

            b = len(all_bonds)
            all_bonds.append((x, y))
            #fbonds.append(torch.cat([fatoms[y], bond_features(bond)], 0))
            fbonds.append(bond_features(bond))
            in_bonds[y].append(b)
            out_bonds[x].append(b)

            b = len(all_bonds)
            all_bonds.append((y, x))
            fbonds.append(bond_features(bond))
            in_bonds[x].append(b)
            out_bonds[y].append(b)

        scope.append((total_atoms, n_atoms))
        total_atoms += n_atoms

    total_bonds = len(all_bonds)
    fatoms = torch.stack(fatoms, 0)
    fbonds = torch.stack(fbonds, 0)
    aoutgraph = torch.zeros(total_atoms, MAX_NB).long()
    aingraph = torch.zeros(total_atoms, MAX_NB).long()
    bgraph = torch.zeros(total_bonds, MAX_NB).long()

    for a in range(total_atoms):
        for i, b in enumerate(out_bonds[a]):
            aoutgraph[a, i] = b
        for i, b in enumerate(in_bonds[a]):
            aingraph[a, i] = b

    for b1 in range(1, total_bonds):
        x, y = all_bonds[b1]
        for i, b2 in enumerate(in_bonds[x]):
            if all_bonds[b2][0] != y:
                bgraph[b1, i] = b2

    return fatoms, fbonds, aoutgraph, bgraph, aingraph, scope, all_bonds

#Node-central Encode
class NMPN(nn.Module):

    def __init__(self, hidden_size, depth):
        super(NMPN, self).__init__()
        self.hidden_size = hidden_size
        self.depth = depth

        self.W_nin = nn.Linear(ATOM_FDIM , hidden_size, bias=False)
        self.W_node = nn.Linear(hidden_size+BOND_FDIM, hidden_size, bias=False)
        #self.W_i = nn.Linear(ATOM_FDIM + BOND_FDIM, hidden_size, bias=False)
        #self.W_h = nn.Linear(hidden_size, hidden_size, bias=False)
        #self.W_o = nn.Linear(ATOM_FDIM + hidden_size, hidden_size)

    def forward(self, mol_graph):
        fatoms, fbonds, aoutgraph, bgraph, aingraph, scope, all_bonds= mol_graph
        fatoms = create_var(fatoms)
        fbonds = create_var(fbonds)
        aoutgraph = create_var(aoutgraph)
        #bgraph = create_var(bgraph)

        h_0 = self.W_nin(fatoms)
        h_0 = nn.ReLU()(h_0)
        h_0 = h_0.t()
        H_n = h_0
        #message = nn.ReLU()(binput)

        for i in range(self.depth):
            #Message function
            message = self.messagefunction(H_n,fbonds,all_bonds)
            nei_message = index_select_ND(message, 0, aoutgraph)
            nei_message = nei_message.sum(dim=1)
            nei_message = self.W_node(nei_message)
            nei_message = nei_message.t()
            #update function
            H_n = nn.ReLU()(h_0 + nei_message)

        return H_n
    def messagefunction(self,H_n,fbonds,all_bonds):
        total_bonds = len(fbonds)
        in_n = []
        for b1 in range(1, total_bonds):
            x, y = all_bonds[b1]
            in_n.append(y)
        in_n = create_var(torch.tensor(in_n))
        message = H_n.index_select(1,in_n)
        message = message.t()
        zero = create_var(torch.unsqueeze(torch.zeros(message.size()[1:]),0))
        message = torch.cat([zero,message],0)
        message = torch.cat([message , fbonds], 1)
        return message

#Edge-central Encoder
class EMPN(nn.Module):

    def __init__(self, hidden_size, depth, out):
        super(EMPN, self).__init__()
        self.hidden_size = hidden_size
        self.depth = depth
        self.out = out
        self.W_ein = nn.Linear(BOND_FDIM , hidden_size, bias=False)
        self.W_edge = nn.Linear(hidden_size+ATOM_FDIM, hidden_size, bias=False)
        self.W_eout = nn.Linear(hidden_size + ATOM_FDIM, out, bias=False)
        #self.W_i = nn.Linear(ATOM_FDIM + BOND_FDIM, hidden_size, bias=False)
        #self.W_h = nn.Linear(hidden_size, hidden_size, bias=False)
        #self.W_o = nn.Linear(ATOM_FDIM + hidden_size, hidden_size)

    def forward(self, mol_graph):
        fatoms, fbonds, aoutgraph, bgraph, aingraph, scope, all_bonds = mol_graph
        fatoms = create_var(fatoms)
        fbonds = create_var(fbonds)
        #aoutgraph = create_var(aoutgraph)
        bgraph = create_var(bgraph)
        aingraph = create_var(aingraph)

        h_0 = self.W_ein(fbonds)
        h_0 = nn.ReLU()(h_0)
        H_e = h_0
        #message = nn.ReLU()(binput)

        for i in range(self.depth):
            # Message function
            message = self.messagefunction(H_e, fatoms, all_bonds)
            nei_message = index_select_ND(message, 0, bgraph)
            nei_message = nei_message.sum(dim=1)
            nei_message = self.W_edge(nei_message)
            H_e = nn.ReLU()(h_0 + nei_message)

        message = self.messagefunction(H_e, fatoms, all_bonds)
        nei_message = index_select_ND(message, 0, aingraph)
        nei_message = nei_message.sum(dim=1)
        nei_message = self.W_eout(nei_message)
        H_e = nn.ReLU()(nei_message)
        H_e = H_e.t()

        return H_e

    def messagefunction(self, H_e, fatoms, all_bonds):
        total_bonds = len(all_bonds)
        out_n = []
        for b1 in range(1, total_bonds):
            x, y = all_bonds[b1]
            out_n.append(x)
        out_n = create_var(torch.tensor(out_n))
        message = fatoms.index_select(0, out_n)
        zero = create_var(torch.unsqueeze(torch.zeros(message.size()[1:]), 0))
        message = torch.cat([zero, message], 0)
        message = torch.cat([H_e,message], 1)
        return message

# class DMPN(nn.Module):
#     def __init__(self, hidden_size, depth, out, atten_size, r,voc_size):
#         super(DMPN, self).__init__()
#         self.hidden_size = hidden_size
#         self.depth = depth
#         self.out = out
#         self.atten_size = atten_size
#         self.r = r
#         self.gru = GRU(voc_size,(self.hidden_size+self.out)*self.r*2)
#         self.NMPN = NMPN(self.hidden_size,self.depth)
#         self.EMPN = EMPN(self.hidden_size,self.depth,self.out)
#         self.W_1 = nn.Linear(self.hidden_size+self.out, self.atten_size, bias=False)
#         self.W_2 = nn.Linear(self.atten_size, self.r, bias=False)
#
#     def forward(self, mol_graph,S_sca,smile_batch):
#         H_n = self.NMPN(mol_graph)
#         H_e = self.EMPN(mol_graph)
#         H_node = torch.cat([H_n, H_e], 0)
#         H_node = H_node.t()
#
#         hidden_space_sca = []
#         hidden_space_side = []
#         for st, le in mol_graph[5]:
#             # readout function
#             s_sca = S_sca[st: st + le]
#             cur_vecs_sca , cur_vecs_side  = self.read_out(H_node[st: st + le] , s_sca)
#             hidden_space_sca.append(cur_vecs_sca)
#             hidden_space_side.append(cur_vecs_side)
#
#         space_sca = torch.stack(hidden_space_sca,0)
#         space_side = torch.stack(hidden_space_side,0)
#         gru_h0 = torch.cat([space_side,space_sca],1)
#         gru_h0 = torch.unsqueeze(gru_h0,0)
#         gru_h0 = gru_h0.repeat([3,1,1])
#         gru_h0 = gru_h0.type(torch.float32)
#         output = self.gru(smile_batch,gru_h0)
#         return output
#
#     def read_out(self, h_node, s_sca):
#         sca_index = []
#         side_index = []
#         for i in range(len(s_sca)):
#             if s_sca[i] == 1:
#                 sca_index.append(i)
#             else:
#                 side_index.append(i)
#         sca_index = create_var(torch.tensor(sca_index))
#         side_index = create_var(torch.tensor(side_index))
#         sca_node = h_node.index_select(0,sca_index)
#         side_node = h_node.index_select(0,side_index)
#         sca_s = F.softmax(self.W_2(nn.Tanh()(self.W_1(sca_node))),1)
#         sca_s = sca_s.t()
#         side_s = F.softmax(self.W_2(nn.Tanh()(self.W_1(side_node))),1)
#         side_s =side_s.t()
#         sca_embeding = torch.flatten(torch.mm(sca_s,sca_node))
#         side_embeding = torch.flatten(torch.mm(side_s,side_node))
#
#         return sca_embeding,side_embeding

class GRU(nn.Module):
    def __init__(self, voc_size,hidden_size):
        super(GRU, self).__init__()
        #self.embedding = nn.Embedding(voc_size, 128)
        self.gru = nn.GRU(voc_size,hidden_size,3)
        self.linear = nn.Linear(hidden_size, voc_size)

    def forward(self, x, h):
        #x = self.embedding(x)
        x ,h_n = self.gru(x,h)
        x = self.linear(x)
        return x

def replace_halogen(string):
    """Regex to replace Br and Cl with single letters"""
    br = re.compile('Br')
    cl = re.compile('Cl')
    string = br.sub('R', string)
    string = cl.sub('L', string)

    return string
def one_hot(ele,lista):
    out = []
    for i in range(len(lista)):
        if lista[i] ==ele:
            out.append(1)
        else:
            out.append(0)
    return out
def cross_entropy(X,H):
    seq_len,batch_size,voc_size = X.size()
    loss = create_var(torch.zeros(batch_size))
    for i in range(seq_len):
        logits = H[i, :, :]
        target = X[i, :, :]
        log_prob = F.log_softmax(logits, dim=1)
        NNLLoss = -torch.sum(log_prob * target,1)
        loss += NNLLoss
    mean_loss = loss.mean()
    return mean_loss

if __name__ == "__main__":

    smiles_batch = SMILES_BATCH
    sca_batch = []
    target = [1,0,0,1]
    target = create_var(torch.tensor(target))
    voc = []
    add_chars = set()
    add_chars.add("pad")
    #获得分子骨架
    for smile in smiles_batch:
        #m = Chem.MolFromSmiles(smile)
        sca = MurckoScaffold.MurckoScaffoldSmilesFromSmiles(smile)
        sca_batch.append(sca)
    # 收集数据中的词汇信息
    for smile in smiles_batch:
        regex = '(\[[^\[\]]{1,6}\])'
        smiles = replace_halogen(smile)
        char_list = re.split(regex, smiles)
        for char in char_list:
            if char.startswith('['):
                add_chars.add(char)
            else:
                chars = [unit for unit in char]
                [add_chars.add(unit) for unit in chars]
    voc = list(add_chars)
    max_length = 50
    j=0
    #tokenize and encode smiles
    for smile in smiles_batch:

        smile_chars = []
        encode = []
        regex = '(\[[^\[\]]{1,6}\])'
        smiles = replace_halogen(smile)
        char_list = re.split(regex, smiles)
        for char in char_list:
            if char.startswith('['):
                smile_chars.append(char)
            else:
                chars = [unit for unit in char]
                [smile_chars.append(unit) for unit in chars]
        for i in range(max_length):
            if i < len(smile_chars):
                encode.append(torch.tensor(one_hot(smile_chars[i], voc)))
            else:
                encode.append(torch.tensor(one_hot("pad", voc)))
        encode_batch = torch.stack(encode,0)
        encode_batch = torch.unsqueeze(encode_batch,1)
        if j == 0:
            encode_last = encode_batch
        else:
            encode_last = torch.cat([encode_last,encode_batch],1)
        j+=1
    encode_last = encode_last.type(torch.float32)
    smiles_batch_encode = create_var(encode_last)
    S_sca = atom_if_sca(smiles_batch,sca_batch)
    mol_graph = mol2graph(smiles_batch)
    # fatoms, fbonds, aoutgraph, bgraph, aingraph, scope ,all_bonds = mol_graph
    hidden_size = 200
    depth = 2
    out = 100
    atten_size = 128
    r =3
    voc_size = len(voc)
    cluster_num =2
    # nmpn = NMPN(hidden_size,depth)
    # nmpn = nmpn.cuda()
    # for param in nmpn.parameters():
    #     if param.dim() == 1:
    #         nn.init.constant(param, 0)
    #     else:
    #         nn.init.xavier_normal(param)
    # H_n = nmpn.forward(mol_graph)
    # print(H_n.shape)
    #
    # empn = EMPN(hidden_size,depth,out)
    # empn = empn.cuda()
    # for param in empn.parameters():
    #     if param.dim() == 1:
    #         nn.init.constant(param, 0)
    #     else:
    #         nn.init.xavier_normal(param)
    # H_e = empn.forward(mol_graph)
    # print(H_e.shape)


    dmpn = DMPN(hidden_size,depth,out,atten_size,r,voc_size,cluster_num)
    dmpn = dmpn.cuda()
    for i in dmpn.named_parameters():
        name , param = i
        gs = ["GaussianMixture.means_","GaussianMixture.covariances_","GaussianMixture.weights_"]
        if name not in gs :
            if param.dim() == 1:
                nn.init.constant(param, 0)
            else:
                nn.init.xavier_normal(param)

        print(name)
        print(param)
    # for param in dmpn.parameters():
    #     if param.dim() == 1:
    #         nn.init.constant(param, 0)
    #     else:
    #         nn.init.xavier_normal(param)

    #loss1 , loss2 = dmpn.cross_loss(smiles_batch, sca_batch, smiles_batch_encode, target)
    result = dmpn.sample(10, smiles_batch, sca_batch, smiles_batch_encode)
    #H_node = dmpn(mol_graph, sca_batch, smiles_batch_encode)
    # H_node = torch.squeeze(H_node,1)
    #
    #loss = result + loss2
    #loss.backward()
    print(result)
    #print(H_node.shape)
    print(target.shape)
