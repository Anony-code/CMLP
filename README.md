# CMLP

This code is implemented based on [CGBL](https://github.com/QueuQ/CGLB/tree/master).

# Requirement

To run the code, the following packages are required to be installed:
 
* python==3.7.10
* scipy==1.5.2
* numpy==1.19.1
* torch==1.7.1
* networkx==2.5
* scikit-learn~=0.23.2
* matplotlib==3.4.1
* ogb==1.3.1
* dgl==0.6.1
* dgllife==0.2.6

# Dataset

We use the same dataset as [CGBL](https://github.com/QueuQ/CGLB/tree/master).

For importing the N-CGL datasets, use the following command in python:

```
from CGLB.NCGL.utils import NodeLevelDataset
dataset = NodeLevelDataset('data_name')
```

where the 'data_name' should be replaced by the name of the selected dataset, which are:

```
'CoraFull-CL'
'Arxiv-CL'
'Reddit-CL'
'Products-CL'
```

# Run Experiments

```
python train.py
```

# TODO
We will update to a more readable version of the code upon acceptance.
