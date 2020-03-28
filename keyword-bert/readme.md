# experiments descriptions
```
network structure of Keyword-BERT:

	  fusion layer & output layer balabala...
	 ---------------------
h(CLS)<= transformer |  keyword attention transformer => h_keyword(AB)
  	 layer  n    |  layer n
  	 ---------------------
  	 transformer layer n-1
  	 ---------------------
  	       ......
  	 ---------------------
  	 transformer layer 1
  	 ---------------------
  	    embedding layer
```

# change list (compared with vanilla BERT)
`model.py`: 
  * Add a keyword attention layer.
  * Add a flexible layer-num configuration

`run_classifier.py`: 
  * Add keyword mask (parsing from training data). 
  * Add a fusion layer. 
  * Add hooks to print training loss and
iteration steps.

# data preparation
Add `keyword mask` to vanilla BERT training data. (plz refer to `data/convert_to_bert_keyword.py` and `pre_tokenize.py`)

For data privacy, we only publish a small sample of our real data.

# training procedure
1. Prepare you own training and test data
2. Copy bert pre-trained Chinese large model to `pre_trained`
3. Modify model hyper-parameters in `pre_trained/bert_config.json`. 
Modify running parameter in `run.sh` (if needed). 
4. Execute `run.sh`

