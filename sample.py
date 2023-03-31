from utils import construct_vocabulary
from data_structs import Vocabulary,MolData
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import DMPN
import argparse
import torch
from utils import valid_smiles
import pandas as pd
import time
from time import strftime
from time import gmtime

# Argument parser
parser = argparse.ArgumentParser(description='Neural message passing and rnn')
parser.add_argument('--vocPath', default='D:\Python\ProjectOne2.0\data\Voc', help='voc path')
parser.add_argument('--modelPath', default='D:\Python\ProjectOne2.0\model\model._009.pt', help='model path')
parser.add_argument('--save_dir', default='D:\Python\ProjectOne2.0\data\sample.csv', help='save sample path')
parser.add_argument('--batch-size', type=int, default=256, metavar='N',help='Input batch size for training ')
parser.add_argument('--epochs', type=int, default=300, metavar='N',help='Number of epochs to sample')
parser.add_argument('--molecule_num', type=int, default=30000, metavar='N',help='sample number')
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
    # define model
    dmpn = DMPN(args.hidden_size, args.depth, args.out, args.atten_size, args.r, args.d_hid, args.d_z, voc)
    dmpn = dmpn.cuda()
    if torch.cuda.is_available():
        dmpn.load_state_dict(torch.load(args.modelPath))
    totalsmiles = set()

    mol = ["O=C(CNC(=O)c1ccco1)OCC(=O)c1ccc2ccccc2c1"]
    sca = ["c1ccc2ccccc2c1"]
    start = time.time()
    all_valid = 0
    all_seq = 0
    for epoch in range(args.epochs):
        seqs,_ = dmpn.sample(args.batch_size,mol,sca)
        valid = 0
        for i, seq in enumerate(seqs.cpu().numpy()):
            smile = voc.decode(seq)
            if valid_smiles(smile):
                valid += 1
                totalsmiles.add(smile)

        molecules_total = len(totalsmiles)
        print("Epoch {}: {} ({:>4.1f}%) molecules were valid. [{} / {}]".format(epoch + 1, valid,
                                                                                100 * valid / len(seqs),
                                                                                molecules_total, args.molecule_num))
        all_valid +=valid
        all_seq +=len(seqs)
        if molecules_total > args.molecule_num:
            break
    print('Sampling completed')
    end = time.time()
    time_spent = strftime("%H:%M:%S", gmtime(end - start))
    print("train time spent {time}".format(time=time_spent))
    print("sample all valid {:>4.1f}%".format(100*all_valid / all_seq))
    print("sample all uniqueness {:>4.1f}%".format(100*molecules_total/all_valid))
    df_smiles = pd.DataFrame()
    df_smiles['Smiles'] = list(totalsmiles)
    df_smiles.head(args.molecule_num).to_csv(args.save_dir, index=None)
    return molecules_total

if __name__ == "__main__":
    main(args)