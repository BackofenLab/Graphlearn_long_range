##################
# ARGPARSE
#################

import argparse
parser = argparse.ArgumentParser(description='generating graphs given few examples')
parser.add_argument('--n_jobs', help='number of jobs')
parser.add_argument('--train_load', help='path to train smiles file')
parser.add_argument('--n_samples', help=' how many smiles to generate')
parser.add_argument('--gen_save', help='where to save generated smiles strings')

args = parser.parse_args()
print('ARGS:',args)



# 1. load smiles for training 
import rdkitutils  as rdk
z=open(args.train_load,'r').read().split()[1:]
z=[zz[:-6] for zz in z ]
graphs = list(rdk.smiles_strings_to_nx(z))[:100]

# 2. train grammar
from graphlearn.lsgg_layered import lsgg_layered
from graphlearn.test.cycler import Cycler
c=Cycler()

decomposition_args={ "base_thickness_list":[2],
                    "radius_list": [0],
                    "thickness_list": [1]}

coarsened = [c.encode_single(g)for g in graphs] # this is the problematic one
grammar = lsgg_layered(decomposition_args=decomposition_args).fit(coarsened)

print("GRAMMAR TRAINED")

# 3. train estimator for samplingn 
from graphlearn.score import OneClassEstimator
esti = OneClassEstimator().fit(graphs)

print("ESTI TRAINED")
# 4 sample 

from graphlearn import sample
from graphlearn.select import SelectMax
import basics as b

def sample_single(x):
    graph, grammar, esti =x 
    return sample.sample(graph,Cycler(),grammar=grammar,scorer=esti, selector=SelectMax())

it = [(g,grammar,esti) for g in graphs[:4]  ]
res = b.mpmap(sample_single,it, poolsize=int(args.n_jobs))


# 5 write to dst
rdk.nx_to_smi(res,args.gen_save)
