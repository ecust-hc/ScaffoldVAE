from utils import  read_smiles_csv,SA,get_mol,QED,logP,NP,Weight
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == '__main__':
    gen = read_smiles_csv("D:\Python\ProjectOne2.0\data\sample.csv")
    ref = ["O=C(CNC(=O)c1ccco1)OCC(=O)c1ccc2ccccc2c1"]
    qed =[]
    sa = []
    logp = []
    np = []
    weight = []

    for i in range(len(gen)):
        qed.append(QED(get_mol(gen[i])))
        sa.append(SA(get_mol(gen[i])))
        logp.append(logP(get_mol(gen[i])))
        np.append(NP(get_mol(gen[i])))
        weight.append(Weight(get_mol(gen[i])))
    gen_similarty = pd.read_csv("D:\Python\ProjectOne2.0\data\sample_similarty.csv")
    similarty = gen_similarty["similarty"].values.tolist()

    plt.subplot(2, 3, 1)
    sns.kdeplot(qed, fill=True, color="g", label='QED')
    plt.legend(loc='upper left',fontsize=6)
    plt.subplot(2, 3, 2)
    sns.kdeplot(sa, fill=True, color="g", label='SA')
    plt.legend(loc='upper left',fontsize=6)
    plt.subplot(2, 3, 3)
    sns.kdeplot(logp, fill=True, color="g", label='logP')
    plt.legend(loc='upper left',fontsize=6)
    plt.subplot(2, 3, 4)
    sns.kdeplot(np, fill=True, color="b", label='NP')
    plt.legend(loc='upper left',fontsize=6)
    plt.subplot(2, 3, 5)
    sns.kdeplot(weight, fill=True, color="b", label='Weight')
    plt.legend(loc='upper left',fontsize=6)
    plt.subplot(2, 3, 6)
    sns.kdeplot(similarty,fill=True, color="b", label='similarty')
    plt.legend(loc='upper left',fontsize=6)
    plt.tight_layout()  # 自动调整子批次参数，使子批次适合图形区域
    plt.show()
    print("ref qed is  {}".format(QED(get_mol(gen[0]))))
    print("ref sa is  {}".format(SA(get_mol(gen[0]))))
    print("ref logp is  {}".format(logP(get_mol(gen[0]))))
    print("ref np is  {}".format(NP(get_mol(gen[0]))))
    print("ref weight is  {}".format(Weight(get_mol(gen[0]))))