import scipy.sparse
import numpy as np
import pandas as pd
import scipy.sparse
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect as Morgan
from multiprocessing import Pool
import torch
import time
import scipy
import h5py

def mapper(n_jobs):
    '''
    Returns function for map call.
    If n_jobs == 1, will use standard map
    If n_jobs > 1, will use multiprocessing pool
    If n_jobs is a pool object, will return its map function
    '''
    if n_jobs == 1:
        def _mapper(*args, **kwargs):
            return list(map(*args, **kwargs))

        return _mapper
    if isinstance(n_jobs, int):
        pool = Pool(n_jobs)

        def _mapper(*args, **kwargs):
            try:
                result = pool.map(*args, **kwargs)
            finally:
                pool.terminate()
            return result

        return _mapper
    return n_jobs.map

def calc_self_tanimoto( s1, s2, device='cpu'):

    # Convert input vectors to PyTorch tensors and move to the specified device
    gen_vecs = s1
    gen_vecs2 = s2
    num , h = gen_vecs.shape
    x_gen = torch.tensor(gen_vecs).to(device).half()
    y_gen = torch.tensor(gen_vecs2).to(device).half()

    # Transpose x_stock tensor
    y_gen = y_gen.transpose(0, 1)

    # Calculate Tanimoto similarity using matrix multiplication
    tp = torch.mm(x_gen, y_gen)
    jac = (tp / (x_gen.sum(1, keepdim=True) + y_gen.sum(0, keepdim=True) - tp))

    # Handle NaN values in the Tanimoto similarity matrix
    jac = jac.masked_fill(torch.isnan(jac), 1)
    #zero = torch.zeros(num).to(device).half()
    #jac = torch.where(jac > 0.6, jac, zero)

    return jac

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

def fingerprint(smiles_or_mol, fp_type='morgan', dtype=None, morgan__r=2, morgan__n=1024):
    """
    Generates fingerprint for SMILES
    If smiles is invalid, returns None
    Returns numpy array of fingerprint bits
    Parameters:
        smiles: SMILES string
        type: type of fingerprint: [MACCS|morgan]
        dtype: if not None, specifies the dtype of returned array
    """
    fp_type = fp_type.lower()
    molecule = get_mol(smiles_or_mol)
    if molecule is None:
        return None
    if fp_type == 'maccs':
        keys = MACCSkeys.GenMACCSKeys(molecule)
        keys = np.array(keys.GetOnBits())
        fingerprint = np.zeros(166, dtype='uint8')
        if len(keys) != 0:
            fingerprint[keys - 1] = 1  # We drop 0-th key that is always zero
    elif fp_type == 'morgan':
        fingerprint = np.asarray(Morgan(molecule, morgan__r, nBits=morgan__n),
                                 dtype='uint8')
    else:
        raise ValueError("Unknown fingerprint type {}".format(fp_type))
    if dtype is not None:
        fingerprint = fingerprint.astype(dtype)
    return fingerprint

def fingerprints(smiles_mols_array, n_jobs=1, already_unique=True, **kwargs):
    '''
    Computes fingerprints of smiles np.array/list/pd.Series with n_jobs workers
    e.g.fingerprints(smiles_mols_array, type='morgan', n_jobs=10)
    Inserts np.NaN to rows corresponding to incorrect smiles.
    IMPORTANT: if there is at least one np.NaN, the dtype would be float
    Parameters:
        smiles_mols_array: list/array/pd.Series of smiles or already computed
            RDKit molecules
        n_jobs: number of parralel workers to execute
        already_unique: flag for performance reasons, if smiles array is big
            and already unique. Its value is set to True if smiles_mols_array
            contain RDKit molecules already.
    '''
    if isinstance(smiles_mols_array, pd.Series):
        smiles_mols_array = smiles_mols_array.values
    else:
        smiles_mols_array = np.asarray(smiles_mols_array)
    if not isinstance(smiles_mols_array[0], str):
        already_unique = True

    if not already_unique:
        smiles_mols_array, inv_index = np.unique(smiles_mols_array, return_inverse=True)

    fps = mapper(n_jobs)(fingerprint, smiles_mols_array)

    length = 1
    for fp in fps:
        if fp is not None:
            length = fp.shape[-1]
            first_fp = fp
            break
    fps = [fp if fp is not None else np.array([np.NaN]).repeat(length)[None, :]
           for fp in fps]
    if scipy.sparse.issparse(first_fp):
        fps = scipy.sparse.vstack(fps).tocsr()
    else:
        fps = np.vstack(fps)
    if not already_unique:
        return fps[inv_index]
    return fps

if __name__ == "__main__":

    smiles_list = []
    with open("./scaffold.smi", 'r') as f:
        for i, line in enumerate(f):
            smiles = line
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                smiles_list.append(Chem.MolToSmiles(mol))
            # if i >1000:
            #     break
    smiles_list = list(set(smiles_list))
    print(len(smiles_list))
    s = time.time()
    smiles_fingerprint = fingerprints(smiles_list,)
    smiles_fingerprint = np.array(smiles_fingerprint)
    len , _= smiles_fingerprint.shape
    index = int(len/2)

    #将矩阵分片求，最开始是使用双层for循环求解相似性，所求affiny matrix 大约 100000*100000的大小，所以占用内存大，时间巨慢。也想过多线程的方法最终失败。
    #后通过在gpu上是同tensor进行矩阵运算，显存不够，所以将这个相似性矩阵分为4块进行计算，但是tensor将其拼接在一起还是会出现显存不够的问题，所以将4块矩阵分别
    #先存起来，使用np.savatxt(),和np.loadtxt(),存储和加载的时间巨慢，所以使用h5py包提供的方法，将四块矩阵进行存储，以及后面拼接，推荐使用numpy 因为tensor
    #要比numpy更占用内存。
    S1 = smiles_fingerprint[0:index+1,:]
    S2 = smiles_fingerprint[index:,:]

    #之前尝试的使用多线程方法
    # data_list = [(S1,S1),(S1,S2),(S2,S1),(S2,S2)]
    # pool = multiprocessing.Pool(2)
    # result = pool.map(calc_self_tanimoto, data_list)
    # pool.close()
    # pool.join()

    #分片计算矩阵
    tani_matrix = calc_self_tanimoto(S1,S1,device= "cuda:0")
    # tani_matrix12 = calc_self_tanimoto(S1, S2, device="cuda:0")
    # tani_matrix21 = calc_self_tanimoto(S2, S1, device="cuda:0")
    # tani_matrix22 = calc_self_tanimoto(S2, S2, device="cuda:0")

    #tensor拼接 显存不够，用numpy在cpu上进行拼接
    # tani_index1 = torch.cat((tani_matrix11,tani_matrix12),1)
    # tani_index2 = torch.cat((tani_matrix21, tani_matrix22), 1)
    # tani_matrix = torch.cat((tani_index1,tani_index2),0)

    matrix = np.array(tani_matrix.cpu())

    #之前想提前计算laplacian矩阵的 发现不可行
    #D = torch.diag(torch.sum(tani_matrix, axis=1))
    #laplacian = D - tani_matrix
    #laplacian = np.array(laplacian.cpu())
    #laplacian = laplacian.astype(np.float32)
    #_, eigenvectors = eigsh(laplacian, k=20, which='SM')

    e = time.time()

    #想通过numpy存储和读取 时间太慢
    #result1 = np.array(tani_matrix)
    #np.savetxt('./npresult_all.txt', result1)
    #matrix = np.array(tani_matrix11.cpu())

    print(matrix.shape)
    with h5py.File('h5file_com4_22_km.h5', 'w') as hf:
        hf.create_dataset('elem', data=matrix, compression='gzip', compression_opts=4)
    print("Use time: {:.4f}s".format(e - s))

    # sc = SpectralClustering(30, affinity='precomputed', n_init=10, assign_labels='discretize')
    # sc.fit_predict(tani_matrix11)
    #
    # with open("./scaffold_cluster_new.smi", "w") as f:
    #     for i in range(len(smiles_list)):
    #         f.write(smiles_list[i] + " " + str(sc.labels_[i]) + '\n')

