

# we just make a folder with all the data...  args(mostly file-paths) will be taken care of by main2.py


# USAGE:
#python3 mosesmkdata.py --neg AID/bursi_neg.smi --pos AID/bursi_pos.smi --testsize 500 --trainsize 100 200 300 400

# 


import main2 as main 
import os 
import rdkitutils as rdk

os.mkdir("newtraindata")

def mk1set(ind ,randseed):
    p,n,ptrain,ntrain = main.get_all_graphs(randseed)
    rdk.nx_to_moses(p,f"ptest_{ind}.csv")
    rdk.nx_to_moses(n,f"ntest_{ind}.csv")
    for pt in ptrain:
        di=f"newtraindata/p{len(pt)}_{ind}"
        os.mkdir(di)
        rdk.nx_to_moses(pt,di+'/train.csv')
    for nt in ntrain:
        di=f"newtraindata/n{len(nt)}_{ind}"
        os.mkdir(di)
        rdk.nx_to_moses(nt,di+'/train.csv')

for i,e in enumerate([1,2,3]):
    mk1set(i,e)

