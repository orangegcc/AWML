# AWML
## Data
We provide FB15k and WN18 datasets used for the visualization and evaluation in the folder data_FB15k/ and data_WN18 respectively, 
using the input format required by our codes.  
FB15k is published by "Translating Embeddings for Modeling Multi-relational Data (2013)." [Download](https://everest.hds.utc.fr/doku.php?id=en:transe)  
WN18 is published by "A Semantic Matching Energy Function for Learning with Multi-relational Data (2012)." [Download](https://github.com/glorotxa/WakaBST)  
The original data can be downloaded from:  
[Download](https://everest.hds.utc.fr/doku.php?id=en:transe)


Datasets are required in the following format, containing 16 files:  
* -train.txt: training file, format (head_entity, relation, tail_entity).
* -valid.txt: validation file, same format as -train.txt
* -test.txt: testing file, same format as -train.txt
* -train-lhs.pkl, -train-rel.pkl, -train-rhs.pkl: training matrices for head, relation and tail respectively.
* -valid-lhs.pkl, -valid-rel.pkl, -valid-rhs.pkl: validation matrices for head, relation and tail respectively.
* -test-lhs.pkl, -test-rel.pkl, -test-rhs.pkl: testing matrices for head, relation and tail respectively.
* entity2idx.pkl or synset2idx.pkl, idx2entity.pkl or synset2idx.pkl: key-value pairs for entity/relations-id.
* entity2id.txt, relation2id.txt: key-value files, format (entity/relation, id)

Please note that, for TransE and TransR, the dataset required by our codes is in the folder data_FB15k/ and data_WN18/,  
while for TransE(AdaGrad), the dataset required by our codes is in the folder AWML_TransEmin/data/.

## Code
The codes are in the folder AWML_TransE/, AWML_TransEmin/, AWML_TransR/. The original model can be downloaded from:  
* TransE, in folder AWML_TransE/, is published by "Translating Embeddings for Modeling Multi-relational Data (2013)." [Download](https://everest.hds.utc.fr/doku.php?id=en:transe)  
* TransE(AdaGrad), in folder AWML_TransEmin/, is published by "Efficient energy-based embedding models for link prediction in knowledge graphs (2016)." [Download](https://github.com/pminervini/ebemkg)  
* TransR, in folder AWML_TransR/, is published by "Learning Entity and Relation Embeddings for Knowledge Graph Completion (2015)." [Download]( https://github.com/mrlyk423/relation_extraction)

### Pre-training and Clustering
For pre-traning, you need to follow the steps below:  
* TransE: call the program FB15k/WN_TransE.py
* TransE(AdaGrad): call the program learn.py for FB15k and wn_learn.py for WN18 to obtain the embeddings in folder fb15k_embeddings/ and in folder wn18_embeddings/ respectively.  
* TransR: call the program FB15k/WN_TransR.py

For clustering, you need to follow the steps below:  
1. call the program best_valid_model.py to obtain the .txt file for the embeddings.
2. call the program run.sh and clustparse.py in folder cluster/ to cluster all the entity-pair offsets for each knowledge category to cunstruct clustered relation set.  
AP clustering algorithm is published by "Clustering by Passing Messages Between Data Points." [Download](https://github.com/thunlp/KB2E/tree/master/cluster)  
Note that, we provide our clustering result in k.pkl file.
3. call the program rel2subrel_apC.py and parse_trainC.py to obtain the clustered training matrices for head, relation and tail for the training of our proposed framework AWML:  
We provide the dictionary of relation to sub-relation in rel2subrel_apC.pkl and subrel2rel_apC.pkl.
* TransE: FB15k-train-inpl/inpo/inpr_C.pkl for FB15k and WN-train-inpl/inpo/inpr_C.pkl for WN18.  
* TransE(AdaGrad): FB15k-train_C.pkl for FB15k and WN-train_C.pkl for WN18 in folder AWML_TransEmin/data/.  
* TransR: FB15k-train-inpl/inpo/inpr_RC.pkl for FB15k and WN-train-inpl/inpo/inpr_RC.pkl for WN18.  

### Training AWML framework
For calculating the category-specific density, you need to follow the steps below:
1. call the program dif_50dim.py to obtain the entity-pair offsets for each knowledge category.  
2. call the program density_rel.py to calculate each category-specific density.  

For training the KRL model incorporated by our proposed framework, you need to call the training program below:
1. TransE: CTransE_aml/awl_random/pretrain.py  
2. TransE(AdaGrad): learnC_aml/awl_random/pretrain.py  
3. TransR: CTransR_aml/awl_random/pretrain.py  

### Testing the model
We provide the embeddings obtained by all the models used for visualization and evaluation in the folder fb15k_embeddings/ and wn18_embeddings/.  
We also provide the parameters of AWML algorithm for the above embedding result in the corresponding training file.  
For testing in the tasks of link prediction and triplet classification, you need to call the program below:  
* Link prediction: relrank_lp.py for filtered setting and relrank_lp_raw.py for raw setting.  
* Triplet classification: relrank_tc.py for filtered setting and relrank_tc_raw.py for raw setting.  
Please note that, for TransE(AdaGrad) model, the testing process follows the training process in the training file.  

We also provide evaluation results .out file for all the models in folder AWML_TransE/, AWML_TransEmin/, AWML__TransR/.

### Visualization
For visualizing the embeddings of entity-pair offsets, you need to follow the steps below:
1. call the program tsne_transe.py to obtain 2-dim vectors of all the entities and relations.  
The dimensionality reduction algorithm of t-SNE is published by "Visualizing Data using t-SNE" [Download](http://ticc.uvt.nl/˜lvdrmaaten/tsne)
2. call the program dif_2dim.py to obtain all the golden entity-pair offsets.
3. call the program dif_2dim_random.py to obtain all the synthetic entity-pair offsets.
4. call the program rel_plot_posneg_random.py to obtain the visualizing results.
