##################
# ARGPARSE
#################

import argparse
parser = argparse.ArgumentParser(description='generating graphs given few examples')
parser.add_argument('--n_jobs',type=int, help='number of jobs')
parser.add_argument('--train_load', help='path to train smiles file')
parser.add_argument('--n_samples',type=int, help=' how many smiles to generate')
parser.add_argument('--gen_save', help='where to save generated smiles strings')
parser.add_argument('-s','--n_select',type=int, help='this many survive each roung')
parser.add_argument('-n','--n_neighbors',type=int, help='this many random neighbors are sampled for each surviving instance')
parser.add_argument('-p','--n_steps', type=int,help='this many rounds are conducted')
parser.add_argument('--mincip',type=int, help='min cip count for grammar training')
args = parser.parse_args()
print('ARGS:',args)



# 1. load smiles for training 
import rdkitutils  as rdk
z=open(args.train_load,'r').read().split()[1:]
z=[zz[:-6] for zz in z ]
graphs = list(rdk.smiles_strings_to_nx(z))

# 2. train grammar
from graphlearn.lsgg_layered import lsgg_layered
from graphlearn.test.cycler import Cycler
c=Cycler()

decomposition_args={ "base_thickness_list":[2],
                    "radius_list": [0.1],
                    "thickness_list": [1]}
filter_args={ "min_cip_count": args.mincip, "min_interface_count": 2}
coarsened = [c.encode_single(g)for g in graphs] # this is the problematic one
grammar = lsgg_layered(decomposition_args=decomposition_args, filter_args=filter_args).fit(coarsened)

print("GRAMMAR TRAINED")

# 3. train estimator for samplingn 
from graphlearn.score import OneClassEstimator
esti = OneClassEstimator().fit(graphs)

print("ESTI TRAINED")

# 4 sample 

from graphlearn import sample
from graphlearn.choice import SelectProbN as SPN
import basics as b

def sample_single(x):
    #print (".",end='')
    graph, grammar, esti, n_select, n_steps, n_neigh=x 
    return sample.multi_sample(graph,Cycler(),grammar=grammar,scorer=esti, selector=SPN(n_select),n_steps=n_steps,n_neighbors=n_neigh)

sel,ste,nei = args.n_select, args.n_steps, args.n_neighbors 
if args.n_samples != None:
    graphs= graphs[:args.n_samples]
it = [(g,grammar,esti,sel,ste,nei) for g in graphs  ]
print('STARTMAP')
res = b.mpmap_prog(sample_single,it,chunksize=1, poolsize=args.n_jobs)

# 5 write to dst
rdk.nx_to_smi(res,args.gen_save)
open(args.gen_save+".args","w").write(str(args))
