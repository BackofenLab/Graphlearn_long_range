import graphlearn as gl
import numpy as np
import graphlearn.lsgg_loco as loco
import graphlearn.lsgg_layered as lsggl 
import graphlearn.lsgg_locolayer as lsgg_LL
from graphlearn.test import cycler
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
import basics.sgexec as sgexec

from rdkit import rdBase
rdBase.DisableLog('rdApp.*')

parser = argparse.ArgumentParser(description='generating graphs given few examples')
parser.add_argument('--n_jobs',type=int, help='number of jobs')
parser.add_argument('--emit',type=int, help='emit every x graphs')
parser.add_argument('--burnin',type=int, help='start emiting after this many steps')
parser.add_argument('--grammar',type=str, help='priosim, ??')
parser.add_argument('--sge', dest='sge', action='store_true')
parser.add_argument('--no-sge', dest='sge', action='store_false')
parser.add_argument('--neg',type=str, help='negative dataset')
parser.add_argument('--pos',type=str, help='positive dataset')
parser.add_argument('--loglevel',type=int,default =0, help='loglevel')
parser.add_argument('--testsize',type=int, help='number of graphs for testing')
parser.add_argument('--n_steps',type=int,default=15, help='how many times we propose new graphs during sampling')
parser.add_argument('--trainsizes',type=int,nargs='+', help='list of trainsizes')
parser.add_argument('--repeatseeds',type=int,default=[1,2,3],nargs='+', help='list of seeds for repeats.. more seeds means more repeats')
parser.add_argument('--radii',type=int,default =[0,1,2],nargs='+', help='radiuslist')
parser.add_argument('--thickness',type=int,default = 1, help='thickness, 1 is best')
parser.add_argument('--min_cip',type=int,default = 1, help='cip min count')
parser.add_argument('--reg',type=float,default = .25 , help='regulates aggressiveness of acepting worse graphs')
args = parser.parse_args()


logging.basicConfig(stream=sys.stdout, level=args.loglevel)

# 1. load a (shuffeled) negative and positive dataset, + cache..
def getnx(fname,randseed=123):

    '''can we load from cache?'''
    cachename = fname+".cache"
    if os.path.isfile(cachename):
        graphs = ba.loadfile(cachename)
    else:
        '''if not load normaly and write a cache'''
        with gzip.open(fname,'rb') as fi:
            smiles = fi.read()
        graphs = list(rut.smiles_strings_to_nx([line.split()[1] for line in  smiles.split(b'\n')[:-1]]))
        ba.dumpfile(graphs,cachename)

    '''shuffle and return'''
    random.seed(randseed)
    random.shuffle(graphs)
    return graphs



def loadsmi(fname,randseed = 123):
    g = list(rut.smi_to_nx(fname))
    random.seed(randseed)
    random.shuffle(g)
    return g


# 2. use Y graphs for validation and an increasing rest for training
def get_all_graphs(randseed = 123):
    pos = loadsmi(args.pos,randseed) if "bursi" in args.pos else getnx(args.pos,randseed)
    neg = loadsmi(args.neg,randseed) if "bursi" in args.neg else getnx(args.neg,randseed)
    print("lenpos:", len(pos))
    print("lenneg:", len(neg))
    ptest,prest= pos[:args.testsize], pos[args.testsize:]
    ntest,nrest= neg[:args.testsize], neg[args.testsize:]
    return ptest,ntest,[prest[:x] for x in args.trainsizes],[nrest[:x] for x in args.trainsizes]



# 3. for each train set (or tupple of sets) generate new graphs 
def classic(graphs):
    grammar = lsgg.lsgg(
            decomposition_args={"radius_list": args.radii, 
                                "thickness_list": [args.thickness]
                               },
            filter_args={"min_cip_count": args.min_cip,                               
                         "min_interface_count": 2}
            ) 
    assert len(graphs) > 10
    grammar.fit(graphs,n_jobs = args.n_jobs)
    scorer = score.OneClassSizeHarmMean(n_jobs=args.n_jobs,
    #scorer = score.OneClassEstimator(n_jobs=args.n_jobs,
            model=svm.OneClassSVM(kernel='linear',gamma='auto')).fit(graphs)
    scorer.n_jobs=1 # demons cant spawn children
    selector = choice.SelectClassic(reg=args.reg) 
    transformer = transformutil.no_transform()
    
    sampler = sample.sampler(
            transformer=transformer, 
            grammar=grammar, 
            scorer=scorer, 
            selector=selector, 
            n_steps=args.n_steps, burnin = args.burnin, emit=args.emit) 
    return sampler.sample_burnin,graphs

def priosim(graphs):
    grammar = loco.LOCO(  
            decomposition_args={"radius_list":args.radii, 
                                "thickness_list": [args.thickness],  
                                "thickness_loco": 2},
            filter_args={"min_cip_count": args.min_cip,                               
                         "min_interface_count": 1}
            ) 
    assert len(graphs) > 10
    grammar.fit(graphs,n_jobs = args.n_jobs)
    scorer = score.OneClassSizeHarmMean(n_jobs=args.n_jobs,
            model=svm.OneClassSVM(kernel='linear',gamma='auto')).fit(graphs)
    scorer.n_jobs=1 # demons cant spawn children
    selector = choice.SelectClassic(reg=args.reg) 
    transformer = transformutil.no_transform()
    
    sampler = sample.sampler(
            transformer=transformer, 
            grammar=grammar, 
            scorer=scorer, 
            selector=selector, 
            n_steps=args.n_steps, burnin = args.burnin, emit=args.emit) 
    return sampler.sample_burnin,graphs

def coarse(graphs):
    # UNTESTED
    grammar = lsggl.lsgg_layered(  
            decomposition_args={"radius_list": [0,1,2], 
                                "thickness_list": [1],  
                                "base_thickness_list": [2]
                                },
            filter_args={"min_cip_count": 1,                               
                         "min_interface_count": 1}
            ) 
    assert len(graphs) > 10
    c= cycler.Cycler()
    grammar.fit(c.encode(graphs),n_jobs = args.n_jobs)
    scorer = score.OneClassSizeHarmMean(n_jobs=args.n_jobs,
            model=svm.OneClassSVM(kernel='linear',gamma='auto')).fit(graphs)
    scorer.n_jobs=1 # demons cant spawn children
    selector = choice.SelectClassic(reg=0) 
    
    sampler = sample.sampler(
            transformer=c, 
            grammar=grammar, 
            scorer=scorer, 
            selector=selector, 
            n_steps=args.n_steps, burnin = args.burnin, emit=args.emit) 
    return sampler.sample_burnin,graphs
def coarseloco(graphs):
    # UNTESTED
    grammar = lsgg_LL.lsgg_locolayer(  
            decomposition_args={"radius_list": [0,1,2], 
                                "thickness_list": [1],  
                                "base_thickness_list": [2],
                                "thickness_loco": 2
                                },
            filter_args={"min_cip_count": 1,                               
                         "min_interface_count": 1}
            ) 
    assert len(graphs) > 10
    c= cycler.Cycler()
    grammar.fit(c.encode(graphs),n_jobs = args.n_jobs)
    scorer = score.OneClassSizeHarmMean(n_jobs=args.n_jobs,
            model=svm.OneClassSVM(kernel='linear',gamma='auto')).fit(graphs)
    scorer.n_jobs=1 # demons cant spawn children
    selector = choice.SelectClassic(reg=0) 
    
    sampler = sample.sampler(
            transformer=c, 
            grammar=grammar, 
            scorer=scorer, 
            selector=selector, 
            n_steps=args.n_steps, burnin = args.burnin, emit=args.emit) 
    return sampler.sample_burnin,graphs



# 4. generate a learning curve
def vectorize(graphs):
    return sp.sparse.vstack(ba.mpmap(eden.vectorize,
        [[g] for g in graphs],poolsize=args.n_jobs))


def getscore(gp, gn,xt,yt): 
    gpp = vectorize(gp)
    gnn = vectorize(gn)
    svc = svm.SVC(gamma='auto').fit( sp.sparse.vstack((gpp,gnn)),
            [1]*gpp.shape[0]+[0]*gnn.shape[0]  ) 
    return  svc.score(xt,yt )


def make_scorer(ptest,ntest):
    X_test= sp.sparse.vstack((vectorize(ptest), vectorize(ntest)))
    y_test= np.array([1]*len(ptest)+[0]*len(ntest))
    return lambda pgraphs,ngraphs: getscore(pgraphs,ngraphs,X_test,y_test)

def evaluate(scorer,ptrains,ntrains,res):
    # print curve
    myscore   = []
    baseline = []
    genscore = []
    for p,n in zip(ptrains,ntrains):
        gp = res.pop()
        gn = res.pop()
        if isinstance(gp[0],list):
            gp = [g for gl in gp for g in gl]
            gn = [g for gl in gn for g in gl]

        myscore.append(scorer(gp+p, gn+n))
        baseline.append(scorer(p,n))
        genscore.append(scorer(gp,gn))
        print('gpl:',len(gp))
    return myscore,baseline,genscore

def learncurve_mp(randseed=123,addgraphs=None): 
    # SET UP VALIDATION SET
    ptest,ntest,ptrains, ntrains = get_all_graphs(randseed)
    scorer = make_scorer(ptest,ntest)


    print("using local cores")
    res = []
    for p,n in zip(ptrains,ntrains):
        res.append(ba.mpmap_prog(*addgraphs(p),
            poolsize = args.n_jobs, chunksize=1))
        res.append(ba.mpmap_prog(*addgraphs(n),
            poolsize = args.n_jobs, chunksize=1))
  
    res = res [::-1] # pop will pop from the end...  
    return evaluate(scorer,ptrains,ntrains,res)


def learncurve(randseed=123,executer=None,addgraphs = None): 
    # SET UP VALIDATION SET
    ptest,ntest,ptrains, ntrains = get_all_graphs(randseed)
    scorer = make_scorer(ptest,ntest)

    # send all the jobs
    for p,n in zip(ptrains,ntrains):
        executer.add_job(*addgraphs(p))
        executer.add_job(*addgraphs(n))

    return scorer,ptrains,ntrains
 


class peacemeal():
    def __init__(self,stuff,split):
        self.stuff = stuff
        self.cnksize = int(len(stuff)/ split)
    def get(self):
        ret = self.stuff[:self.cnksize]
        self.stuff = self.stuff[self.cnksize:]
        return ret


if __name__ == "__main__":
    addgraphs =  eval(args.grammar)

    if not args.sge:
        a,b,c = list(zip(*[learncurve_mp(x,addgraphs) for x in args.repeatseeds]))
    else:
        executer = sgexec.sgeexecuter(loglevel=args.loglevel, die_on_fail=False)
        z= [learncurve(x,executer,addgraphs) for x in args.repeatseeds]
        res = executer.execute() 
        peacemeal=peacemeal(res,len(args.repeatseeds))
        a,b,c = list(zip(*[evaluate(s,pt,nt,peacemeal.get()) for s,pt,nt in z ]))

    cm =[ np.mean(x) for x in list(zip(*a))  ] 
    om =[ np.mean(x) for x in list(zip(*b))  ] 
    gm=[ np.mean(x) for x in list(zip(*c))  ] 
    cs=[ np.std(x) for x in list(zip(*a))  ] 
    os=[ np.std(x) for x in list(zip(*b))  ] 
    gs=[ np.std(x) for x in list(zip(*c))  ] 

    print('combined',      cm )
    print('originals only',om )
    print('generated only',gm )
    print('combined',      cs )
    print('originals only',os )
    print('generated only',gs )
    ba.dumpfile([args.trainsizes,(cm,om,gm),(cs,os,gs)],"res.pickle")

