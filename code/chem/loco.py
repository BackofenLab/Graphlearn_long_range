import graphlearn as gl
import numpy as np
import graphlearn.lsgg_loco as loco
import graphlearn.lsgg as lsgg
import graphlearn.score as score
import graphlearn.choice as choice
import graphlearn.test.transformutil as transformutil
import graphlearn.sample as sample
import basics as ba # should use this to sample later 
from functools import partial
import sklearn.svm as svm 
import scipy as sp
import argparse
import random
import eden.graph as eden
import rdkitutils as rut 
import gzip
import os.path
import logging
import sys
import sgexec
logging.basicConfig(stream=sys.stdout, level=5)

from rdkit import rdBase
rdBase.DisableLog('rdApp.*')


#################
# this was once learncurve/main2.py .,, we remodel it so that it simply samples 
# once lile longrange.py
##################

parser = argparse.ArgumentParser(description='generating graphs given few examples')
parser.add_argument('--n_jobs',type=int, help='number of jobs')
parser.add_argument('--n_samples',type=int, help='select this many as seeds discard rest')
#parser.add_argument('--sge',type=bool,default=False, help='normal multiprocessing or sungridengine')
parser.add_argument('--sge', dest='sge', action='store_true')
parser.add_argument('--no-sge', dest='sge', action='store_false')

parser.add_argument('--train_load',type=str, help=' dataset')
parser.add_argument('--gen_save',type=str, help='genereated smiles goes here')
parser.add_argument('--n_steps',type=int,default=15, help='how many times we propose new graphs during sampling')
parser.add_argument('--size_score_penalty',type=float,default=0.0, help='percentage of points reduced for each node that a graph is too large')
args = parser.parse_args()


########
#  get the train data 
######
import rdkitutils  as rdk
z=open(args.train_load,'r').read().split()[1:]
z=[zz[:-6] for zz in z ]
graphs = list(rdk.smiles_strings_to_nx(z))



#############
#  do the training stuff    TODO  
#############

# 3. for each train set (or tupple of sets) generate new graphs 
def addgraphs(graphs):
    #grammar = loco.LOCO(  
    grammar = lsgg.lsgg(
            decomposition_args={"radius_list": [0,1,2], 
                                "thickness_list": [1,2],  
                                "loco_minsimilarity": .8, 
                                "thickness_loco": 4},
            filter_args={"min_cip_count": 1,                               
                         "min_interface_count": 1}
            ) 
    grammar.fit(graphs,n_jobs = args.n_jobs)
    #scorer = score.OneClassEstimator(n_jobs=args.n_jobs).fit(graphs)
    scorer = score.OneClassAndSizeFactor(n_jobs=args.n_jobs,model=svm.OneClassSVM(kernel='linear',gamma='auto')).fit(graphs)
    scorer.n_jobs=1 # demons cant spawn children
    #selector = choice.SelectProbN(1)
    selector = choice.SelectClassic(reg=0) # linear kernel -> .8 , rbf kernel -> .97? 
    transformer = transformutil.no_transform()
    # multi sample: mysample = partial(sample.multi_sample, transformer=transformer, grammar=grammar, scorer=scorer, selector=selector, n_steps=20, n_neighbors=200) 
    # mysample = partial(sample.sample, transformer=transformer, grammar=grammar, scorer=scorer, selector=selector, n_steps=20) 
    mysample = partial(sample.sample_sizeconstraint,penalty=args.size_score_penalty,
            transformer=transformer, 
            grammar=grammar, 
            scorer=scorer, 
            selector=selector, 
            n_steps=args.n_steps) 
    
    #print (mysample(graphs[0]))
    #exit()
    return mysample,graphs




#############
#  somehow save the outputs in readable format 
#################
