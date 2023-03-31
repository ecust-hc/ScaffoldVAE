from utils import construct_vocabulary
from data_structs import Vocabulary,MolData,Experience
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import DMPN
import argparse
import torch
from utils import valid_smiles,unique,ext_sca,create_var
import pandas as pd
import numpy as np
import time
from time import strftime
from time import gmtime
from model import DMPN
import os
from shutil import copyfile
from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit import DataStructs
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Neural message passing and rnn')
parser.add_argument('--vocPath', default='D:\Python\ProjectOne2.0\data\Voc', help='voc path')
parser.add_argument('--restore_prior', default='D:\Python\ProjectOne2.0\model\model._099.pt', help='model path')
parser.add_argument('--restore_agent', default='D:\Python\ProjectOne2.0\model\model._099.pt', help='model path')
parser.add_argument('--save_dir', default='D:\Python\ProjectOne2.0\\agent_model\\agent_model.ckpt', help='save sample path')
parser.add_argument('--batch-size', type=int, default=256, metavar='N',help='Input batch size for training ')
parser.add_argument('--steps', type=int, default=300, metavar='N',help='Number of step')
parser.add_argument('--sigma', type=int, default=20, metavar='N',help='sigma')
parser.add_argument('--molecule_num', type=int, default=3000, metavar='N',help='sample number')
parser.add_argument('--d_z', type=int, default=128, metavar='N',help='z  size')
parser.add_argument('--d_hid', type=int, default=256, metavar='N',help='DMPN model hidden size')
parser.add_argument('--hidden-size', type=int, default=200, metavar='N',help='NMPN , EMPN model hidden size')
parser.add_argument('--depth', type=int, default=3, metavar='N',help='NMPN , EMPN model Hidden vector update times')
parser.add_argument('--out', type=int, default=100, metavar='N',help='EMPN model the size of output')
parser.add_argument('--atten_size', type=int, default=128, metavar='N',help='DMPN model the size of graph attention readout')
parser.add_argument('--r', type=int, default=3, metavar='N',help=' r different insights of node importance')
args = parser.parse_args()
print(args)

def main(args):
    voc = Vocabulary(init_from_file='data/Voc')
    start_time = time.time()
    Prior = DMPN(args.hidden_size, args.depth, args.out, args.atten_size, args.r, args.d_hid, args.d_z, voc)
    Agent = DMPN(args.hidden_size, args.depth, args.out, args.atten_size, args.r, args.d_hid, args.d_z, voc)
    Prior = Prior.cuda()
    Agent = Agent.cuda()
    if torch.cuda.is_available():
        Prior.load_state_dict(torch.load(args.restore_prior))
        Agent.load_state_dict(torch.load(args.restore_agent))
    for param in Prior.parameters():
        param.requires_grad = False
    optimizer = torch.optim.Adam(Agent.parameters(), lr=0.0005)
    experience = Experience(voc)
    experience_replay = 0
    step_score = [[], []]
    print("Model initialized, starting training...")
    mol = ["O=C(CNC(=O)c1ccco1)OCC(=O)c1ccc2ccccc2c1"]
    sca = ["c1ccc2ccccc2c1"]
    for step in range(args.steps):
        seqs, agent_likelihood = Agent.sample(args.batch_size, mol, sca)
        unique_idxs = unique(seqs)
        seqs = seqs[unique_idxs]
        agent_likelihood = agent_likelihood[unique_idxs]
        seqs_list = seqs.cpu().numpy().tolist()
        seqs_batch = []
        seqs_list_len = len(seqs_list)
        agent_likelihood_list = agent_likelihood.cpu().numpy().tolist()
        smiles_batch =[]
        likelihood_list = []
        for i, seq in enumerate(seqs_list):
            smile = voc.decode(seq)
            if valid_smiles(smile) and smile not in smiles_batch:
                likelihood_list.append( agent_likelihood_list[i])
                seqs_batch.append(seq)
                smiles_batch.append(smile)
        valid = len(smiles_batch)/seqs_list_len
        mol_batch,sca_batch,agent_likelihood,encode_batch = ext_sca(smiles_batch,likelihood_list,seqs_batch)
        agent_likelihood = create_var(torch.tensor(agent_likelihood))
        _,_,prior_likelihood = Prior.forward(mol_batch,sca_batch,create_var(torch.tensor(encode_batch)))
        score = scoring_function(mol_batch)
        augmented_likelihood = prior_likelihood + args.sigma * create_var(score)
        loss = torch.pow((augmented_likelihood - agent_likelihood), 2)
        if experience_replay and len(experience)>4:
            exp_seqs, exp_score, exp_prior_likelihood = experience.sample(4)
            exp_agent_likelihood, exp_entropy = Agent.likelihood(exp_seqs.long())
            exp_augmented_likelihood = exp_prior_likelihood + args.sigma * exp_score
            exp_loss = torch.pow((create_var(exp_augmented_likelihood) - exp_agent_likelihood), 2)
            loss = torch.cat((loss, exp_loss), 0)
            agent_likelihood = torch.cat((agent_likelihood, exp_agent_likelihood), 0)

        prior_likelihood = prior_likelihood.data.cpu().numpy()
        new_experience = zip(mol_batch, score, prior_likelihood)
        experience.add_experience(new_experience)
        loss = loss.mean()
        # Add regularizer that penalizes high likelihood for the entire sequence
        loss_p = - (1 / agent_likelihood).mean()
        loss += 5 * 1e3 * loss_p

        # Calculate gradients and make an update to the network weights
        loss.requires_grad_(True)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Convert to numpy arrays so that we can print them
        augmented_likelihood = augmented_likelihood.data.cpu().numpy()
        agent_likelihood = agent_likelihood.data.cpu().numpy()

        # Print some information for this step
        time_elapsed = (time.time() - start_time) / 3600
        time_left = (time_elapsed * ((args.steps - step) / (step + 1)))
        print("\n       Step {}   Fraction valid SMILES: {:4.1f}  Time elapsed: {:.2f}h Time left: {:.2f}h".format(
            step, valid * 100, time_elapsed, time_left))
        #print("  Agent    Prior   Target   Score             SMILES")
        #for i in range(10):
        #    print(" {:6.2f}   {:6.2f}  {:6.2f}  {:6.2f}     {}".format(agent_likelihood[i],
        #                                                               prior_likelihood[i],
        #                                                               augmented_likelihood[i],
        #                                                               score[i],
        #                                                               mol_batch[i]))

        # Need this for Vizard plotting
        step_score[0].append(step + 1)
        step_score[1].append(np.mean(score))

        #experience.print_memory(os.path.join(args.save_dir, "memory.smi"))
        torch.save(Agent.state_dict(), args.save_dir[:-4]+'_{0:03d}.pt'.format(step))
    plt.plot(step_score[0], step_score[1])
    plt.show()

def scoring_function(smiles):
    scores = []
    k = 0.7
    query_structure = "O=C(CNC(=O)c1ccco1)OCC(=O)c1ccc2ccccc2c1"
    query_mol = Chem.MolFromSmiles(query_structure)
    query_fp = AllChem.GetMorganFingerprint(query_mol, 2, useCounts=True, useFeatures=True)
    for i in range(len(smiles)):
        mol = Chem.MolFromSmiles(smiles[i])
        if mol:
            fp = AllChem.GetMorganFingerprint(mol, 2, useCounts=True, useFeatures=True)
            score = DataStructs.TanimotoSimilarity(query_fp, fp)
            score = min(score, k) / k
            scores.append(float(score))
    return np.array(scores)

if __name__ == "__main__":
    main(args)