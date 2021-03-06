odels =['aae','char_rnn','vae','organ']
#models =['char_rnn','vae'] # organ fails to run

repeats = [1,2,3]


#m this newtraindata was generated by the moses datamaker and then linked to the moses 
# so moses wrote in there.. 
path = 'moses_lc2_2'
import rdkitutils as rdk
import numpy as np
import main2 as main
import sys
import basics as ba
import scipy as sp

def lc(model='aae',rep=0):
    p= ba.loadfile(f"{path}/ptest_{rep}.pick")
    n= ba.loadfile(f"{path}/ntest_{rep}.pick")
    scorer = main.make_scorer(p,n) 
    p=rdk.moses_to_nx(f"{path}/ptrain_{rep}.csv")
    n=rdk.moses_to_nx(f"{path}/ntrain_{rep}.csv")


    combined =[]
    base =[]
    newonly=[]
    opALL = list(rdk.smi_to_nx(f"{path}/ptrain_{rep}/{model}/gen"))
    onALL = list(rdk.smi_to_nx(f"{path}/ntrain_{rep}/{model}/gen"))
    for size in main.args.trainsizes: 
        op=opALL[:size]
        on=onALL[:size]
        f= lambda a,b : (main.vectorize(a+b), [1]*len(a)+[0]*len(b))
        newonly.append(scorer(f(op,on)))
        #base.append(scorer(f(p,n)))
        #combined.append(scorer(f(p+op,n+on)))
    return combined,base,newonly 


model=main.args.model
print (model)
a,b,c = list(zip(*[lc(model,x) for x in repeats]))
#main.format_abc(a,b,c,f"{model}.pickle") 


