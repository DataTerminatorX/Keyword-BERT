# experiments descriptions
```
network structure of Keyword-BERT:

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

# change list (compare to raw BERT)
`extract_features.py`: Add keyword mask parsing from training data

`model.py`: Add keyword attention layer.

`run_classifier.py`: Add fusion layer. Add hooks to print training loss and 
iteration steps.

# data preparation
Remember to add `keyword mask` to raw BERT training data, plz refer to `pre_tokenize.py` and
 `data` folder for more details

For data privacy, we only publish a small sample of our real data.

# training procedure
1. Prepare you own training and test data
2. Copy bert pre-trained Chinese large model to `pre_trained`
3. Modify model hyper-parameters in `pre_trained/bert_config.json`. 
Modify running parameter in `run.sh` (if needed). 
4. Execute `run.sh`

