

import argparse
parser = argparse.ArgumentParser(description='generating graphs given few examples')
parser.add_argument('--n_jobs',type=int, help='number of jobs')
parser.add_argument('--neg',type=str, help='negative dataset')
parser.add_argument('--pos',type=str, help='positive dataset')
parser.add_argument('--testsize',type=int, help='number of graphs for testing')
parser.add_argument('--trainsize',type=int,nargs='+', help='list of trainsizes')

args = parser.parse_args()
print('ARGS:',args)


# 1. load a negative and positive dataset and shuffle 

import rdkitutils as rut 
p = [ x for x in rut.smi_to_nx(args.pos)]
print(len(p))

#with gzip.open('119_active.txt.gz','rb') as fil:
#    for line in fi:
        #print('got line', line.split()[1])


# 2. use Y graphs for validation and an increasing rest for training

# 3. for each train set (or tupple of sets) generate new graphs 

# 4. generate a learning curve


