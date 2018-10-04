# AWML
## Data
We provide FB15k and WN18 datasets used for the visualization and evaluation in the folder data_FB15k/ and data_WN18 respectively, 
using the input format required by our codes.  
These two datasets are published by "Translating Embeddings for Modeling Multi-relational Data (2013)."  
The original dataset can be downloaded from <https://everest.hds.utc.fr/doku.php?id=en:transe>, containing five files in the following format:  

Datasets are required in the following format, containing 16 files:  
* -train.txt: training file, format (head_entity, relation, tail_entity).
* -valid.txt: validation file, same format as -train.txt
* -test.txt: testing file, same format as -train.txt
* -train-lhs.pkl, -train-rel.pkl, -train-rhs.pkl: training matrices for head, relation, tail respectively.
* -valid-lhs.pkl, -valid-rel.pkl, -valid-rhs.pkl: validation matrices for head, relation, tail respectively.
* -test-lhs.pkl, -test-rel.pkl, -test-rhs.pkl: testing matrices for head, relation, tail respectively.
