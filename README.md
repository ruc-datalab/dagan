# Adaptive Data Augmentation for Supervised Learning over Missing Data

## Framework
![avatar](https://github.com/ruclty/dagan/blob/master/figs/architecture.jpg)

## Requirements
Before running the code, please make sure your Python version is above **3.6**.
Then install necessary packages by :
```sh
pip3 install -r requirements.txt
```

## Parameters
 You need to write a .json file as the configuration. The keyworks should include :

 - name: required, name of the output file 
 - source: required, path of the source data 
 - target: required, path of the target data 
 - target_mask: required, path of the target mask matrix
 - gen_model: required, neural network of the generator
 - normalize_cols: required, index of the numerical attributes normalized by simple-normalization 
 - gmm_cols: required, index of the numerical attributes normalized by GMM-normalization 
 - one-hot_cols: required, index of the categorical attributes encoded by one-hot encoding 
 - ordinal_cols:mrequired, index of the categorical attributes encoded by ordinal encoding 
 - epochs: required, number of training epochs 
 - steps_per_epoch: required, steps per epoch
 - rand_search: required, whether to search hyper-parameters randomly, yes or no ,
 - param: required if rand_search is 'no', hyper-parameter of the NN  

Folder "code/params" contains examples, you can run the code using those parameter files directly, or write a self-defined parameter file to train a new dataset.

## Run
Run the code with the following command :
```sh
python code/train.py [parameter file] [gpu_id]
```
A example running command:
```sh
python code/train.py code/params/param-eyestate-MNAR 0
```
