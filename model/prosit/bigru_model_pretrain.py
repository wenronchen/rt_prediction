#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 23:25:16 2020

@author: wenrchen
"""



import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import keras
import sys

import time

from scipy.stats import pearsonr
def Pearson(act, pred):
    return pearsonr(act, pred)[0]


def Delta_t95(act, pred):
    num95 = int(np.ceil(len(act) * 0.95))
    return 2 * sorted(abs(act - pred))[num95 - 1]

def Delta_tr95(act, pred):
    return Delta_t95(act, pred) / (max(act) - min(act))

def MSE(act,pred,norm_rt):
    act=np.transpose(np.matrix(act))
    pred=np.transpose(np.matrix(pred))
    return np.square((act/norm_rt-pred/norm_rt)).mean()

#from keras.utils import plot_model
from matplotlib import pyplot as plt
from keras.initializers import RandomUniform, VarianceScaling
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input, Dense, Embedding, GRU, Dropout, Bidirectional,LeakyReLU
import layers
import preprocess_seq_for_prosit
from keras.callbacks import EarlyStopping

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

TRAIN_EPOCHS = 5
TRAIN_BATCH_SIZE = 16
PRED_BATCH_SIZE = 16
PRED_BAYES = False
PRED_N = 100

VAL_SPLIT = 0.8

dropout_rate=0.5
embed_out=24
GRU_units=256
dense_feature=256
dense_dropout=0.1

data_path=sys.argv[1]
output=sys.argv[2]
max_seq=int(sys.argv[3])
max_rt=float(sys.argv[4])
min_rt=float(sys.argv[5])

GRU_untis=int(sys.argv[6])
embed_out=int(sys.argv[7])
dense_feature=int(sys.argv[8])
dropout_rate=float(sys.argv[9])
dense_dropout=float(sys.argv[10])

y_attr=sys.argv[11]

pretrained=int(sys.argv[12])
pretrain_path=sys.argv[13]




def build_model(input_length):
    peptides_in=Input([input_length,],name='peptides_in')
    
    embedding=Embedding(input_dim=21,output_dim=embed_out,embeddings_initializer=RandomUniform(maxval=0.05,minval=-0.05))(peptides_in)
    
    encoder1=Bidirectional(GRU(int(GRU_units/2), kernel_initializer=VarianceScaling(scale=1, mode='fan_avg', distribution='uniform'),\
                               return_sequences=True))(embedding)
    
    dropout_1=Dropout(dropout_rate)(encoder1)
    
    encoder2=GRU(GRU_units,kernel_initializer=VarianceScaling(scale=1, mode='fan_avg', distribution='uniform'),return_sequences=True)(dropout_1)
    
    dropout_2=Dropout(dropout_rate)(encoder2)
    
    encoder_att=layers.Attention()(dropout_2)
    
    pep_dense1=Dense(dense_feature,activation='relu',kernel_initializer=VarianceScaling(scale=1, mode='fan_avg', distribution='uniform'))(encoder_att)
    
    pep_dense1_lReLu=LeakyReLU(alpha=0.30000001192092896)(pep_dense1)
    
    pep_dense1_drop=Dropout(dense_dropout)(pep_dense1_lReLu)
    
    prediction=Dense(1,activation='linear',use_bias=True,kernel_initializer=VarianceScaling(scale=1, mode='fan_avg', distribution='uniform'))(pep_dense1_drop)
    
    model=Model(inputs=[peptides_in],outputs=[prediction])
    
    model.compile(loss='mean_squared_error',optimizer=Adam(lr=1e-3))

    print(model.summary())
    
    #plot_model(model,to_file='./data/RPLC_result/plots/model_bigru.png')
    
    return model

def PlotLosses(history,path):
  plt.subplot()  
  plt.plot(history.history['loss'])  
  plt.plot(history.history['val_loss'])  
  plt.title('model loss')  
  plt.ylabel('loss')  
  plt.xlabel('epoch')  
  plt.legend(['train', 'test'], loc='upper left')  
  #plt.show()
  plt.savefig(path+'_loss.png')
  
def Plot_RT(df,path,norm_rt,min_rt,y_attr):
    norm_rt=norm_rt/60
    min_rt=min_rt/60
    
    act=df[y_attr].values
    pred=df['predict'].values
    
    act=np.asarray(act)
    act=act/60
    pred=np.asarray(pred)
    pred=pred*(norm_rt-min_rt)
    
    pearson=Pearson(act,pred)
    delta_t95=Delta_t95(act,pred)
    delta_tr95=Delta_tr95(act,pred)
    r2=r2_score(act,pred)
    mse=MSE(act,pred,norm_rt)
    
    print("pearson = ",pearson,"delta_t95 = ",delta_t95,"delta_tr95 = ",delta_tr95,"r_squared = ",r2,"MSE = ",mse)
    
    plt.title("prediction result")
    plt.grid(True)
    plt.style.use('seaborn-whitegrid')


    plt.scatter(act,pred,color='red')
    lims=np.linspace(min_rt,norm_rt,2000)
    plt.plot(lims,lims,linestyle='--',color='blue',label='reference')
    
    plt.legend() #show the label
    
    plt.xlim([min_rt,norm_rt])
    plt.ylim([min_rt,norm_rt])
    plt.xlabel('observed ')
    plt.ylabel('predicted')

    #plt.show()
    plt.savefig(path+'_result.png')

    
    

def train_with_df(model,max_seq,max_rt,min_rt,y_attr,data_type,data_path,output_path):
    
#    train_output=output_path+"_train.tsv"
#    test_output=output_path+"_test.tsv"

#    loss="mean_squared_error"
#    optimizer = "Adam"
    
    #find the maximum retention time from the unsplit dataset
    #df_combined=pd.read_csv(data_path+'.tsv',sep='\t')
    
    #max_rt=0.0
    #for i in range(df_combined.shape[0]):
    #    if(df_combined.iloc[i]['Retention time']>max_rt):
    #        max_rt=df_combined.iloc[i]['Retention time']
    print(max_rt)
    
    if(data_type=="top-down"):
        x_train,y_train,df_train = preprocess_seq_for_prosit.load_dataset(data_path,max_seq,max_rt,min_rt,'Proteoform',y_attr)
        #x_test,y_test,df_test = preprocess_seq_for_prosit.load_dataset(test_path,max_seq,max_rt,min_rt,'Proteoform',y_attr)
    else:
        x_train,y_train,df_train = preprocess_seq_for_prosit.load_dataset(data_path,max_seq,max_rt,min_rt,'sequence','RT')
        #x_test,y_test,df_test = preprocess_seq_for_prosit.load_dataset(test_path,max_seq,max_rt,min_rt,'sequence','RT')
    print(x_train.shape)
    print(max(y_train),min(y_train))
    

#        yaml_file = open('~/rt/data/RPLC_data/model_irt_prediction/model.yaml', 'r')
#        loaded_model_yaml = yaml_file.read()
#        yaml_file.close()
#        
#        from keras.models import model_from_yaml
#        model = model_from_yaml(loaded_model_yaml)
    model_begin=time.time()    
    model.load_weights(pretrain_path)
    print('>> note: load pre-trained model weights from:',pretrain_path)
    
    #model.compile(optimizer=optimizer, loss=loss)
    #K.set_value(model.optimizer.lr, 0.0001)
#    es=EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
#    history=model.fit(
#        x=x_train,
#        y=y_train,
#        epochs=TRAIN_EPOCHS,
#        batch_size=TRAIN_BATCH_SIZE,
#        validation_data=(x_test,y_test),
#        shuffle="batch",
#        callbacks=[es]
##       callbacks=callbacks,
#    )
#    PlotLosses(history,output_path)
    #plot_model(model,to_file='./data/RPLC_result/plots/model_prosit.png')
    model_end=time.time()
    print("--- %s seconds ---" % (model_end - model_begin))
    predict_begin=time.time()
    predict_train=model.predict(x_train)
    predict_end=time.time()
    print("--- %s seconds ---" % (predict_end - predict_begin))
    
    df_train.insert(2,'predict',predict_train)
    loss_train=predict_train
    
    for i in range(0,predict_train.shape[0]):
      loss_train[i]=predict_train[i]-y_train[i]
        
    df_train.insert(3,'prediction_loss',loss_train)
    print("Result of train set: \n")
    Plot_RT(df_train,output_path,max_rt,min_rt,y_attr)
    df_train.to_csv(output_path+'.tsv',sep='\t',index=None)
    
    
#    predict_test=model.predict(x_test)
#    
#    df_test.insert(2,'predict',predict_test)
#    loss_test=predict_test
#    
#    for i in range(0,predict_test.shape[0]):
#      loss_test[i]=predict_test[i]-y_test[i]
#        
#    df_test.insert(3,'loss',loss_test)
#    print("Result of test set: \n")
#    Plot_RT(df_test,output_path+'_test',max_rt,min_rt,y_attr)
#    df_test.to_csv(test_output,sep='\t',index=None)
    
#    model_yaml = model.to_yaml()
#    with open(output+"_model.yaml", "w") as yaml_file:
#        yaml_file.write(model_yaml)
#    # serialize weights to HDF5
#    model.save_weights(output+"_model_weights.h5")
#    print("Saved model to disk.")
    
    keras.backend.get_session().close()


#preprocess_seq_for_prosit.load_dataset(data_path+"_train.tsv",max_seq,max_rt,'Proteoform','experimental_mobility')
#preprocess_seq_for_prosit.load_dataset(data_path+"_test.tsv",max_seq,max_rt,'Proteoform','experimental_mobility')
model=build_model(max_seq)

train_with_df(model,max_seq,max_rt,min_rt,y_attr,"top-down",data_path,output)


