# Graphlearn with long range dependencies

We adapt a graph grammar to incorporate long range dependencies.
This is acchieved by coarsening a graph and combining 
the interface on the base and coarsened level. 


## Experimental results

We test out grammar on RNA molecules. 
Generated graphs are assigned a bit-score by a state of the art oracle.
Human expers have determines that an instance with a bitscore of higher
than 39 should be considered to belong to the RNA family in question. 

<img src="performance.png">
<img src="similarity.png">


## Usage General Case

This should take care of all dependencies.
```
pip3 install git+https://github.com/fabriziocosta/EDeN.git --user
pip3 install graphlearn structout
```

Lets first load some data, input musst be networkx graphs with 'label' annotation on every node/edge.
The cycler will generate a coarsened graph that contracts cycles 
and store the original graph in g.graphp['original'].

```python3
from graphlearn.util import util as util_top
from graphlearn.test import cycler
g = util_top.test_get_circular_graph()
gplus=g.copy()
gplus.node[0]['label']='weird' 
c=cycler.Cycler()
g=c.encode_single(g)
gplus = c.encode_single(gplus)
```



Fitting a grammar and conducting one substitution
```python3
from graphlearn.lsgg_layered import lsgg_layered
decomposition_args={ "base_thickness_list":[2],
                    "radius_list": [0],
                    "thickness_list": [1]}
lsggg = lsgg_layered(decomposition_args=decomposition_args)
lsggg.fit([g, gplus, g,gplus])
neigh = lsggg.neighbors(gplus).__next__()
```

Print original and generated graph to terminal
```python3
import structout as so
so.gprint([gplus, gplus.graph['original']], size =15)
neigh=c.encode_single(neigh)
so.gprint([neigh, neigh.graph['original']], size =15)
```

## Usage RNA

The code below demonstrates how to use this technique on RNAs, 
if you are interested in the application on general graphs please see the
modern implementation showcased above.



```python
import graphlearn.abstract_graphs.RNA as rna
from graphlearn.estimator import Wrapper as estimatorwrapper

sampler=rna.AbstractSampler(radius_list=[0,1], 
                            thickness_list=[2],  
                            min_cip_count=1, 
                            min_interface_count=2, 
                            preprocessor=rna.PreProcessor(base_thickness_list=[1],ignore_inserts=True),
                            postprocessor=rna.PostProcessor(),
                            estimator=estimatorwrapper( nu=.5, cv=2, n_jobs=-1)
                           )
sequences = sampler.sample([sequences from an RNA family])

```


## Graph grammar 

A modern version of the grammar can be found at 
https://github.com/fabriziocosta/GraphLearn


## Running the experiments

Experiments are in code/experiments.
HPC scripts are for the "sun grid engine".

## comparison to GraphRNN

In the paper we demonstrated GraphRNN applied to RNA grpahs. Our modified GraphRNN:

https://github.com/smautner/GraphRNN



# MOSES TEST

| Meassure     | aae     | char\_rnn | vae       | organ     | baseline   | coarsened  |
|--------------|---------|----------|-----------|-----------|------------|------------|
| valid        | 0.604   | 0.351    | 0.001     | 0.108     | 1.0        | 1.0        |
| unique@1000  | 0.358   | 1.0      | 1.0       | 0.944     | 1.0        | 1.0        |
| unique@10000 | 0.358   | 1.0      | 1.0       | 0.944     | 1.0        | 1.0        |
| FCD/Test     | 33.985  | 6.136    | nan       | 33.185    | 34.845     | 35.778     |
| SNN/Test     | 0.468   | 0.369    | 0.090     | 0.266     | 0.204      | 0.217      |
| Frag/Test    | 0.908   | 0.982    | 0.0       | 0.558     | 0.919      | 0.893      |
| Scaf/Test    | 0.0480  | 0.265    | nan       | 0.0       | 0.0        | 0.0        |
| FCD/TestSF   | 35.092  | 6.949    | nan       | 33.994    | 35.994     | 36.377     |
| SNN/TestSF   | 0.448   | 0.364    | 0.091     | 0.261     | 0.202      | 0.215      |
| Frag/TestSF  | 0.905   | 0.975    | 0.0       | 0.558     | 0.910      | 0.891      |
| Scaf/TestSF  | 0.0     | 0.022    | nan       | 0.0       | 0.0        | 0.0        |
| IntDiv       | 0.695   | 0.849    | 0.0       | 0.847     | 0.855      | 0.829      |
| IntDiv2      | 0.659   | 0.836    | 0.0       | 0.808     | 0.850      | 0.824      |
| Filters      | 0.957   | 0.934    | 1.0       | 0.583     | 0.079      | 0.872      |
| logP         | 0.0146  | 0.0839   | 8.752     | 0.388     | 1.186      | 3.260      |
| SA           | 0.311   | 0.081    | 0.266     | 0.438     | 11.714     | 9.176      |
| QED          | 0.002   | 0.0002   | 0.185     | 0.05145   | 0.315      | 0.279      |
| NP           | 0.306   | 0.099    | 5.090     | 2.971     | 4.025      | 3.325      |
| weight       | 323.785 | 52.685   | 76736.756 | 24619.107 | 66388.0129 | 140922.662 |
