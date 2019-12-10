

import argparse
import random
import eden.graph as eden
parser = argparse.ArgumentParser(description='generating graphs given few examples')
parser.add_argument('--n_jobs',type=int, help='number of jobs')
parser.add_argument('--neg',type=str, help='negative dataset')
parser.add_argument('--pos',type=str, help='positive dataset')
parser.add_argument('--testsize',type=int, help='number of graphs for testing')
parser.add_argument('--trainsizes',type=int,nargs='+', help='list of trainsizes')
args = parser.parse_args()
print('ARGS:',args)


# 1. load a negative and positive dataset and shuffle 

import rdkitutils as rut 
import gzip
import os.path

def getnx(fname):
    
    cachename = fname+".cache"
    if os.path.isfile(cachename):
        print("read from cache")
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
import graphlearn as gl
import graphlearn.lsgg_loco as loco
import graphlearn.score as score
import graphlearn.choice as choice
import graphlearn.test.transformutil as transformutil
import graphlearn.sample as sample
import basics as ba # should use this to sample later 
def addgraphs(graphs):
    
    grammar = loco.LOCO(  
            decomposition_args={"radius_list": [0,1,2], 
                                "thickness_list": [1],  
                                "loco_minsimilarity": .8, 
                                "thickness_loco": 4},
            filter_args={"min_cip_count": 1,                               
                         "min_interface_count": 1}
            ) 
    scorer = score.OneClassEstimator().fit(graphs)
    selector = choice.SelectMaxN(20)
    transformer = transformutil.no_transform()

    return graphs + [sample.multi_sample(graph,transformer,grammar,scorer,selector) for graph in graphs  ] 


# 4. generate a learning curve
import sklearn.svm as svm 
import scipy as sp
def learncurve(): 
    ptest,ntest,ptrains, ntrains = get_all_graphs()
    
    X_test= sp.sparse.hstack( (eden.vectorize(ptest), eden.vectorize(ntest))), 
    y_test= [1]*len(ptest)+[0]*len(ntest)

    for p,n in zip(ptrains,ntrains):
        pgraphs = eden.vectorize(addgraphs(p))
        ngraphs = eden.vectorize(addgraphs(n))
        svc = svm.SVC().fit( sp.sparse.hstack(pgraphs,ngraphs),[1]*len(pgraphs)+[0]*len(ngraphs)  ) 
        score = svc.score(X_test,y_test )
        print(score)



learncurve()


