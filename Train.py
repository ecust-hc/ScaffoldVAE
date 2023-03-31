from utils import construct_vocabulary,create_var,decrease_learning_rate,KLAnnealer
from data_structs import Vocabulary,MolData
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import DMPN
import torch.nn as nn
import argparse
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
from time import strftime
from time import gmtime
import os.path

# Argument parser
parser = argparse.ArgumentParser(description='Neural message passing and rnn')
parser.add_argument('--datasetPath', default='D:\Python\ProjectOne2.0\data\mol_sca.smi', help='dataset path')
parser.add_argument('--save_dir', default='D:\Python\ProjectOne2.0\model\model.ckpt', help='save model path')
parser.add_argument('--voc_dir', default='D:\Python\ProjectOne2.0\data\Voc', help='voc path')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',help='Input batch size for training ')
parser.add_argument('--hidden-size', type=int, default=200, metavar='N',help='NMPN , EMPN model hidden size')
parser.add_argument('--d_hid', type=int, default=256, metavar='N',help='DMPN model hidden size')
parser.add_argument('--d_z', type=int, default=128, metavar='N',help='z  size')
parser.add_argument('--depth', type=int, default=3, metavar='N',help='NMPN , EMPN model Hidden vector update times')
parser.add_argument('--out', type=int, default=100, metavar='N',help='EMPN model the size of output')
parser.add_argument('--atten_size', type=int, default=128, metavar='N',help='DMPN model the size of graph attention readout')
parser.add_argument('--r', type=int, default=3, metavar='N',help=' r different insights of node importance')
parser.add_argument('--epochs', type=int, default=20, metavar='N',help='Number of epochs to train (default: 50)')
parser.add_argument('--lr_decrease_rate', type=float, default=0.03, metavar='LR',help='Initial learning rate (default: 1e-4)')
parser.add_argument('--learning_rate', type=float, default=1e-4, metavar='LR',help='Initial learning rate (default: 1e-4)')
parser.add_argument('--kl_w_end', type=float, default=0.05, metavar='kl',help='kl weight')
parser.add_argument('--beta', type=float, default=0.1,help='refers to a hyper-parameter of balancing two losses.')
args = parser.parse_args()
print(args)

def main(args):

    if os.path.isfile(args.voc_dir):
        voc = Vocabulary(init_from_file='data/Voc')
    else:
        print("Construct vocabulary")
        voc_chars = construct_vocabulary(args.datasetPath)
        voc = Vocabulary(init_from_file='data/Voc')

    #Create a Dataset from foles
    print("create dataset")
    moldata = MolData(args.datasetPath, voc)
    data = DataLoader(moldata, batch_size=args.batch_size, shuffle=True, drop_last=True,
                      collate_fn=MolData.collate_fn)
    #define model
    dmpn = DMPN(args.hidden_size, args.depth, args.out, args.atten_size, args.r, args.d_hid, args.d_z, voc)
    dmpn = dmpn.cuda()
    # for param in dmpn.parameters():
    #     if param.dim() == 1:
    #         nn.init.constant_(param, 0)
    #     else:
    #         nn.init.xavier_normal_(param)
    optimizer = torch.optim.Adam(dmpn.parameters(), lr=args.learning_rate)
    start = time.time()
    loss_plt = []
    step_plt = []
    x =0
    kl_annealer = KLAnnealer(args.epochs,args.kl_w_end)
    for epoch in range(0, args.epochs):
        dmpn.train()
        # kl_weight = kl_annealer(3)
        for step, batch in tqdm(enumerate(data), total=len(data)):

            mol_batch,sca_batch,collated_arr = batch
            seq = collated_arr.long()
            kl_loss ,recon_loss,_ = dmpn.forward(mol_batch,sca_batch,seq)
            loss =args.beta*kl_loss + recon_loss

            # Calculate gradients and take a step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if epoch % 2 == 0 and step == 1:
                decrease_learning_rate(optimizer, decrease_by= args.lr_decrease_rate)
            if step % 100 == 0 and step != 0:
                tqdm.write("*" * 50)
                tqdm.write("Epoch {:3d}   step {:3d}    loss: {:5.2f} kl_loss: {:5.2f} recon_loss: {:5.2f} \n".format(epoch, step, loss.item(),kl_loss.item(), recon_loss.item()))
                loss_plt.append(loss.item())
                step_plt.append(x)
                x+=1
        torch.save(dmpn.state_dict(), args.save_dir[:-4]+'_{0:03d}.pt'.format(epoch))
    end = time.time()
    time_spent = strftime("%H:%M:%S", gmtime(end - start))
    print("train time spent {time}".format(time=time_spent))
    plt.title('Train loss vs. epoches', fontsize=20)
    plt.plot(step_plt, loss_plt)
    plt.savefig('data/batch_loss.png')
    plt.show()

if __name__ == "__main__":
    main(args)