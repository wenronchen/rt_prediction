# rt_prediction

## Machine learning models for retention time and migration time prediction

### 1. Prerequisite
Python 3.X and corresponding version of Sklearn, Pytorch, Keras and Tensorflow.

### 2. Scripts to reproduce the results

#### 2.1 Machine learning model results 
To reproduce the results in the paper, go to the corresponding folder in the folder of "model".
For example, the FNN model with LC dataset,

    cd  model/fnn
    sh  fnn_train_ot.sh $1=input $2=output

Similarly, to get the results of CZE dataset, 

    sh  fnn_train_sw480.sh $1=input $2=output
    
#### 2.2 Transfer learning results 
To reproduce the transfer learning results in the paper, for example, For prosit model,

    cd  model/prosit
    sh  prosit_ot_with_pretrain.sh $1=input $2=output $3=pretrained_model

#### 3. Use your own datasets
With your own dataset, you are supposed to format the input file with the following information:

        Proteoform                                             RT/MT
    SLSTFQQMWISKQEYDESGPSIVHRKCF                              6664.46
    APSRKFFVGGNWKMNGRKQSLGELIGTLNAAKVPADTEVV                  6736.14
    TAKTEWLDGKHVVFGKVKEGMNIVEAMERFGSRNGKTSKKITIADCGQLE        6834.56
Then follow the format of bash scripts for models and get your own results. 
  

