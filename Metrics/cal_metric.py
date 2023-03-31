from multiprocessing import Pool
from utils import mapper, valid_smiles, read_smiles_csv, get_mol, canonic_smiles, compute_scaffolds
import argparse
import pandas as pd
from utils import canonicalize_smiles_from_file

def fraction_valid(gen, n_jobs=1):
    """
    Computes a number of valid molecules
    Parameters:
        gen: list of SMILES
        n_jobs: number of threads for calculation
    """
    gen = mapper(n_jobs)(valid_smiles, gen)
    return gen.count(True) / len(gen)

def remove_invalid(gen, canonize=True, n_jobs=1):
    """
    Removes invalid molecules from the dataset
    """
    if not canonize:
        mols = mapper(n_jobs)(get_mol, gen)
        return [gen_ for gen_, mol in zip(gen, mols) if mol is not None]
    return [x for x in mapper(n_jobs)(canonic_smiles, gen) if
            x is not None]

def fraction_unique(gen, n_jobs=1, check_validity=True):
    """
    Computes a number of unique molecules
    Parameters:
        gen: list of SMILES
        n_jobs: number of threads for calculation
        check_validity: raises ValueError if invalid molecules are present
    """
    canonic = set(mapper(n_jobs)(canonic_smiles, gen))
    if None in canonic and check_validity:
        canonic.remove(None)
    return len(canonic) / len(gen)

def scaffold_novelty(gen, train, n_jobs=1):
    # Create the set to store the unique scaffolds
    gen_scaffolds = set(compute_scaffolds(gen, n_jobs=n_jobs))
    train_scaffolds = set(compute_scaffolds(train, n_jobs=n_jobs))

    # Calculate the Scaffold Novelty Score
    scaffold_novelty_score = len(gen_scaffolds - train_scaffolds) / len(gen)

    return scaffold_novelty_score

def scaffold_diversity(gen, n_jobs=1):
    # Create a set to store the unique scaffolds
    scaffolds = compute_scaffolds(gen, n_jobs=n_jobs)

    # Calculate the Scaffold Diversity Score
    scaffold_diversity_score = len(scaffolds) / len(gen)

    return scaffold_diversity_score

def get_all_metrics(gen, n_jobs=1,
                    device='cpu', batch_size=512, pool=None, train=None):
    metrics = {}
    # Start the process at the beginning and avoid repeating the process
    close_pool = False
    if pool is None:
        if n_jobs != 1:
            pool = Pool(n_jobs)
            close_pool = True
        else:
            pool = 1
    metrics['Validity'] = fraction_valid(gen, n_jobs=pool)
    gen_valid = remove_invalid(gen, canonize=True)
    metrics['Uniqueness'] = fraction_unique(gen, pool)
    metrics['Diversity'] = scaffold_diversity(gen_valid, n_jobs=pool)
    if train is not None:
        metrics['Novelty'] = scaffold_novelty(gen_valid, train, n_jobs=pool)

    if close_pool:
        pool.close()
        pool.join()
    return metrics  # , df_distribution

def main(args):
    gen = read_smiles_csv(args.gen_path)
    train = None
    if args.train_path is not None:
        train, _ = canonicalize_smiles_from_file(args.train_path)

    metrics = get_all_metrics(gen=gen, n_jobs=args.n_jobs,
                              device=args.device, train=train,)

    if args.print_metrics:
        for key, value in metrics.items():
            print('{}, {:.4}'.format(key, value))
        return metrics
    else:
        return metrics

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path',default='D:\Python\ProjectOne2.0\data\mol_sca.smi',type=str, required=False,help='Path to train molecules csv')
    parser.add_argument('--gen_path',default='D:\Python\ProjectOne2.0\data\sample.csv',type=str, required=False,help='Path to generated molecules csv')
    parser.add_argument('--output',default='D:\Python\ProjectOne2.0\Metrics\metrics.csv',type=str, required=False,help='Path to save results csv')
    parser.add_argument('--print_metrics', action='store_true', default=True,help="Print results of metrics or not? [Default: False]")
    parser.add_argument('--n_jobs',type=int, default=1,help='Number of processes to run metrics')
    parser.add_argument('--device',type=str, default='cpu',help='GPU device id (`cpu` or `cuda:n`)')
    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    metrics = main(args)
    table = pd.DataFrame([metrics]).T
    table.to_csv(args.output, header=False)