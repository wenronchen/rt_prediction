# rt_prediction

## Machine learning models for retention time and migration time prediction

### 1. Prerequisite
Python 3.X and corresponding version of Sklearn, Pytorch, Keras and Tensorflow.
### 2. Datasets
Two top-down datasets and two bottom-up datasets for pretraining are included in the respository.
#### 2.1 LC data sets
- LC-ONE: The data set was generated from ovarian tumor samples with top-down MS. It contains 610 proteoforms (473 entries for training, 173 entries for testing).
- LC-TEN: The data set was combined with 10 replicates of ovarian tumor data set. It contains 1010 proteoforms (764 entries for training, 274 entries for testing).
- LC-PEPTIDE: The data set was generated from 24 human cell lines and tissues with bottom-up MS. It contains 146, 587 peptides in total. 

#### 2.2 CZE data sets
- CZE-ONE: The data set was generated from SW480 cell lines with top-down MS. It contains 1230 proteoforms (981 entries for training, 249 entries for testing).
- CZE-ALL: The data set was combined with 3 replicates of SW480 data set and 3 replicates of SW620 data set. It contains 2914 proteoforms (2105 entries for training, 809 entries for testing).
- CZE-PEPTIDE: The data set was generated from HeLa cell lines with bottom-up MS. It contains 4234 peptides in total. 

### 3. Scripts to reproduce the results

#### 3.1 Machine learning model results 
To reproduce the results in the paper, go to the corresponding folder in the folder of "model".
For example, the FNN model with LC dataset,

    cd  model/fnn
    sh  fnn_train_LC.sh $1=input $2=output

Similarly, to get the results of CZE dataset, 

    sh  fnn_train_CZE.sh $1=input $2=output
    
#### 3.2 Transfer learning results 
To reproduce the transfer learning results in the paper, for example, For prosit model,

    cd  model/prosit
    sh  prosit_LC_with_pretrain.sh $1=input $2=output $3=pretrained_model

#### 4. Use your own datasets
With your own dataset, you are supposed to format the input file with the following information:

        Proteoform                                             RT/MT norm
    SLSTFQQMWISKQEYDESGPSIVHRKCF                               0.617
    APSRKFFVGGNWKMNGRKQSLGELIGTLNAAKVPADTEVV                   0.624
    TAKTEWLDGKHVVFGKVKEGMNIVEAMERFGSRNGKTSKKITIADCGQLE         0.632
Then follow the format of bash scripts for models and get your own results. 
  

