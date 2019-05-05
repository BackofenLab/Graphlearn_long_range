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


## Usage 

The code below demonstrates how to use this technique on RNAs, 
if you are interested in the application on general graphs please see the
modern implementation.



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

