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

parser = argparse.ArgumentParser(description='generating graphs given few examples')
parser.add_argument('--n_jobs',type=int, help='number of jobs')
parser.add_argument('--neg',type=str, help='negative dataset')
parser.add_argument('--pos',type=str, help='positive dataset')
parser.add_argument('--testsize',type=int, help='number of graphs for testing')
parser.add_argument('--trainsizes',type=int,nargs='+', help='list of trainsizes')
args = parser.parse_args()



# 1. load a (shuffeled) negative and positive dataset, + cache..
def getnx(fname):
    cachename = fname+".cache"
    if os.path.isfile(cachename):
        return ba.loadfile(cachename)
    with gzip.open(fname,'rb') as fi:
        smiles = fi.read()
    atomz = list(rut.smiles_strings_to_nx([line.split()[1] for line in  smiles.split(b'\n')[:-1]]))
    random.seed(123)
    random.shuffle(atomz)
    ba.dumpfile(atomz,cachename)
    return atomz


# 2. use Y graphs for validation and an increasing rest for training
def get_all_graphs():
    pos = getnx(args.pos)
    neg = getnx(args.neg)
    ptest,prest= pos[:args.testsize], pos[args.testsize:]
    ntest,nrest= neg[:args.testsize], neg[args.testsize:]
    return ptest,ntest,[prest[:x] for x in args.trainsizes],[nrest[:x] for x in args.trainsizes]


# 3. for each train set (or tupple of sets) generate new graphs 
def addgraphs(graphs):
    #grammar = loco.LOCO(  
    grammar = lsgg.lsgg(
            decomposition_args={"radius_list": [0,1,2], 
                                "thickness_list": [1],  
                                "loco_minsimilarity": .8, 
                                "thickness_loco": 4},
            filter_args={"min_cip_count": 2,                               
                         "min_interface_count": 2}
            ) 
    grammar.fit(graphs)
    scorer = score.OneClassEstimator().fit(graphs)
    selector = choice.SelectMaxN(10)
    transformer = transformutil.no_transform()
    mysample = partial(sample.multi_sample, transformer=transformer,grammar=grammar,scorer=scorer,selector=selector,n_steps=5) 
    #res  = ba.mpmap_prog(mysample,graphs[:int(len(graphs))],poolsize=10,chunksize=1)
    res = sgexec.sgexec(mysample,graphs)
    return graphs + res 


# 4. generate a learning curve
def learncurve(): 
    # SET UP VALIDATION SET
    ptest,ntest,ptrains, ntrains = get_all_graphs()
    X_test= sp.sparse.vstack((eden.vectorize(ptest), eden.vectorize(ntest)))
    y_test= np.array([1]*len(ptest)+[0]*len(ntest))
    
    # GENERATE ESTIMATORS AND VALIDATE
    for p,n in zip(ptrains,ntrains):
        pgraphs = eden.vectorize(addgraphs(p))
        ngraphs = eden.vectorize(addgraphs(n))
        svc = svm.SVC(gamma='auto').fit( sp.sparse.vstack((pgraphs,ngraphs)),[1]*pgraphs.shape[0]+[0]*ngraphs.shape[0]  ) 
        score = svc.score(X_test,y_test )
        print(score)

learncurve()


