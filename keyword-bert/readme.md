# experiments descriptions
```
network structure of keyword-bert:

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


# data preparation

Remember to add `keyword mask` to raw BERT training data, plz refer to `data` folder for more details
For data privacy, we only publish a small sample of our real data.

# training procedure
1. Prepare you own training data
2. Copy bert pre-trained Chinese large model to `pre_trained`
3. Modify hyper-parameters in `run.sh` and then run it

