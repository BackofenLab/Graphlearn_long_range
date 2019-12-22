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
parser.add_argument('--sge',type=bool,default=False, help='normal multiprocessing or sungridengine')
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



def loadsmi(fname):
    g = list(rut.smi_to_nx(fname))
    random.seed(1337)
    random.shuffle(g)
    return g

    


# 2. use Y graphs for validation and an increasing rest for training
def get_all_graphs():
    pos = loadsmi(args.pos)
    neg = loadsmi(args.neg)
    ptest,prest= pos[:args.testsize], pos[args.testsize:]
    ntest,nrest= neg[:args.testsize], neg[args.testsize:]
    return ptest,ntest,[prest[:x] for x in args.trainsizes],[nrest[:x] for x in args.trainsizes]


# 3. for each train set (or tupple of sets) generate new graphs 
def addgraphs(graphs):
    #grammar = loco.LOCO(  
    grammar = lsgg.lsgg(
            decomposition_args={"radius_list": [0,1], 
                                "thickness_list": [2],  
                                "loco_minsimilarity": .8, 
                                "thickness_loco": 4},
            filter_args={"min_cip_count": 2,                               
                         "min_interface_count": 2}
            ) 
    grammar.fit(graphs,n_jobs = args.n_jobs)
    scorer = score.OneClassEstimator(n_jobs=args.n_jobs).fit(graphs)
    scorer.n_jobs=1 # demons cant spawn children
    #selector = choice.SelectMaxN(10)
    selector = choice.SelectProbN(3)
    transformer = transformutil.no_transform()
    mysample = partial(sample.multi_sample, transformer=transformer,grammar=grammar,scorer=scorer,selector=selector,n_steps=30,n_neighbors=200) 
    return mysample,graphs+graphs+graphs


# 4. generate a learning curve

def vectorize(graphs):
    return sp.sparse.vstack(ba.mpmap(eden.vectorize  ,[[g] for g in graphs],poolsize=args.n_jobs))


from sgexec import sgeexecuter as sge


def getscore(gp, gn,xt,yt): 
    gpp = vectorize(gp)
    gnn = vectorize(gn)
    svc = svm.SVC(gamma='auto').fit( sp.sparse.vstack((gpp,gnn)),[1]*gpp.shape[0]+[0]*gnn.shape[0]  ) 
    return  svc.score(xt,yt )

def learncurve(): 
    # SET UP VALIDATION SET
    ptest,ntest,ptrains, ntrains = get_all_graphs()
    #X_test= sp.sparse.vstack((eden.vectorize(ptest), eden.vectorize(ntest)))
    X_test= sp.sparse.vstack((vectorize(ptest), vectorize(ntest)))
    y_test= np.array([1]*len(ptest)+[0]*len(ntest))
    scorer = lambda x,y: getscore(x,y,X_test,y_test)

    # send all the jobs
    executer = sge()
    for p,n in zip(ptrains,ntrains):
        executer.add_job(*addgraphs(p))
        executer.add_job(*addgraphs(n))
    # execute all the jobs
    res = executer.execute()[::-1] # reverse because pop will pop from the end
    

    # print curve
    for p,n in zip(ptrains,ntrains):
        gp = res.pop()
        gn = res.pop()
        score  = scorer(gp+p, gn+n)
        score2  = scorer(p,n)
        print(score, score2, [len(x) for x in [gp,gn,p,n]])

learncurve()

