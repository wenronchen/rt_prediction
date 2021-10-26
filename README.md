# rt_prediction

## Machine learning models for retention time and migration time prediction

#### 1. Prerequisite
Python 3.X and corresponding version of Pytorch, Keras and Tensorflow.

#### 2. Scripts to reproduce the results
### 2.2 Other
### 2.2 Deep learning models
To reproduce the results in the paper, go to the corresponding folder in the folder of "model".
For example, the FNN model with LC dataset,
    sh  fnn_train_ot.sh $1=input $2=output

Similarly, to get the results of CZE dataset, 
    sh  fnn_train_sw480.sh $1=input $2=output

  

