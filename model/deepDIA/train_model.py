#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 15:05:39 2021

@author: wenrchen
"""

from sklearn.metrics import r2_score
import sys

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
from sklearn.model_selection import GroupKFold
import keras
import numpy as np
import pandas as pd
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import Bidirectional
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping

TRAIN_EPOCHS = 200 #200
TRAIN_BATCH_SIZE = 16
PRED_BATCH_SIZE = 16
PRED_BAYES = False
PRED_N = 100

data_path=sys.argv[1]
output=sys.argv[2]
max_seq=int(sys.argv[3])
max_rt=float(sys.argv[4])
min_rt=float(sys.argv[5])

filter_size=int(sys.argv[6])#64
kernel_size=int(sys.argv[7])#5
lstm_feature=int(sys.argv[8])#128
dropout_rate=float(sys.argv[9])#0.5
dense_feature=int(sys.argv[10])

y_attr=sys.argv[11]

def build_model(max_sequence_length, aa_size=20):
    model = Sequential()
    model.add(
        Conv1D(
            filters=filter_size,
            kernel_size=kernel_size,
            activation="relu",
            input_shape=(max_sequence_length, aa_size)))            
    model.add(MaxPooling1D(pool_size=2, strides=2))
    model.add(Bidirectional(LSTM(lstm_feature, return_sequences=True)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(dense_feature*2, activation='relu'))
    model.add(Dense(dense_feature, activation='relu'))
    model.add(Dense(1, activation='relu'))
    model.compile(
        #loss="mean_absolute_error",
        loss="mean_squared_error",
        optimizer="adam",
        #metrics=["mean_absolute_error"])
        metrics=["mean_squared_error"])
    
    print(model.summary())
    return model

def PlotLosses(history,fold_cnt,path):
  plt.subplot()  
  plt.plot(history.history['loss'])  
  plt.plot(history.history['val_loss'])  
  plt.title('model loss')  
  plt.ylabel('loss')  
  plt.xlabel('epoch')  
  plt.legend(['train', 'test'], loc='upper left')  
  #plt.show()
  plt.savefig(path+'_fold_'+str(fold_cnt)+'_loss.png')
  plt.close()
  
def Plot_RT(df,path,norm_rt,y_attr):
    norm_rt=norm_rt/60
    df=df.copy()
    act=df[y_attr].values
    pred=df['predict'].values
    
    act=np.asarray(act)
    act=act/60
    pred=np.asarray(pred)
    pred=pred*norm_rt
    
    pearson=Pearson(act,pred)
    delta_t95=Delta_t95(act,pred)
    delta_tr95=Delta_tr95(act,pred)
    r2=r2_score(act,pred)
    mse=MSE(act,pred,norm_rt)
    
    print("pearson = ",pearson,"delta_t95 = ",delta_t95,"delta_tr95 = ",delta_tr95,"r_squared = ",r2,"MSE = ",mse)
    
    plt.title("Time prediction")
    plt.grid(True)
    plt.style.use('seaborn-whitegrid')


    plt.scatter(act,pred,color='red')
    lims=np.linspace(0,norm_rt,2000)
    plt.plot(lims,lims,linestyle='--',color='blue',label='reference')
    
    plt.legend() #show the label
    
    plt.xlim([0,norm_rt])
    plt.ylim([0,norm_rt])
    plt.xlabel('observed ')
    plt.ylabel('predicted ')

    #plt.show()
    plt.savefig(path+'_result.png')

def seq_to_tensor(sequences,max_sequence_length,aa_size=20):
    def aa_to_vector(aa):
        vec = np.zeros(aa_size, dtype=int)
        vec['ARNDCEQGHILKMFPSTWYV'.index(aa)] = 1
        return vec

    def seq_to_tensor(seq):
        return [aa_to_vector(aa) for aa in seq]

    return pad_sequences(
        [seq_to_tensor(seq) for seq in sequences],
        maxlen=max_sequence_length,
        padding='post')

def per_re(x,max_rt):
    x=x/max_rt
    return x
def normalize_RT(RTs,max_rt):
    normalized_rt=RTs
    for i in range(len(RTs)):
        normalized_rt[i]=per_re(normalized_rt[i],max_rt)
    return normalized_rt
    

def get_input_output(path,max_seq,max_rt,x_attr,y_attr):
    
    df_data=pd.read_csv(path,sep='\t')
    df=df_data.copy()
    sequences=list(df[x_attr].values)
    RTs=list(df[y_attr].values)
    
    X=seq_to_tensor(sequences,max_seq)
    y=normalize_RT(RTs,max_rt)
    y=np.asarray(y)
    
    return X,y,df_data

def Plot_RT_with_fold(norm_rt,min_rt,fold_cnt,act,pred,path):
    norm_rt=norm_rt/60
    min_rt=min_rt/60
    
    act=np.asarray(act)
    act=act*(norm_rt-min_rt)
    pred=np.asarray(pred)
    pred=pred*(norm_rt-min_rt)
    
    pearson=Pearson(act,pred)
    r2=r2_score(act,pred)
    mse=MSE(act,pred,norm_rt)
    delta_tr95=Delta_tr95(act,pred)
    
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
    plt.savefig(path+'_fold_'+str(fold_cnt)+'_result.png')
    plt.close()
    return pearson,r2,mse,delta_tr95  

def train_with_df(max_seq,max_rt,y_attr,train_path,test_path,output_path):
    
    train_output=output_path+"_train.tsv"
    test_output=output_path+"_test.tsv"

    print(max_rt)
    
    model=build_model(max_seq)
    
    
    x_train,y_train,df_train = get_input_output(train_path,max_seq,max_rt,'Proteoform',y_attr)
    x_test,y_test,df_test = get_input_output(test_path,max_seq,max_rt,'Proteoform',y_attr)

    print(x_train.shape)
    print(max(y_train),min(y_train))
    
    #model.compile(optimizer=optimizer, loss=loss)
    #K.set_value(model.optimizer.lr, 0.0001)
    es=EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=30)
    history=model.fit(
        x=x_train,
        y=y_train,
        epochs=TRAIN_EPOCHS,
        batch_size=TRAIN_BATCH_SIZE,
        validation_data=(x_test,y_test),
        shuffle="batch",
        callbacks=[es]
#       callbacks=callbacks,
    )
    PlotLosses(history,1,output_path)
    #plot_model(model,to_file='./data/RPLC_result/plots/model_prosit.png')
    
    predict_train=model.predict(x_train)
    
    df_train.insert(2,'predict',predict_train)
    loss_train=predict_train
    
    for i in range(0,predict_train.shape[0]):
      loss_train[i]=predict_train[i]-y_train[i]
        
    df_train.insert(3,'prediction_loss',loss_train)
    print("Result of train set: \n")
    Plot_RT(df_train,output_path+'_train',max_rt,y_attr)
    df_train.to_csv(train_output,sep='\t',index=None)
    
    
    predict_test=model.predict(x_test)
    
    df_test.insert(2,'predict',predict_test)
    loss_test=predict_test
    
    for i in range(0,predict_test.shape[0]):
      loss_test[i]=predict_test[i]-y_test[i]
        
    df_test.insert(3,'prediction_loss',loss_test)
    print("Result of test set: \n")
    Plot_RT(df_test,output_path+'_test',max_rt,y_attr)
    df_test.to_csv(test_output,sep='\t',index=None)
    
    

def k_fold_train_with_df(max_seq,max_rt,min_rt,y_attr,data_path,output_path):

    X,y,df = get_input_output(data_path,max_seq,max_rt,'Proteoform',y_attr)


    print(X.shape)
    print(max(y),min(y))
    
    groups=np.asarray(df['Protein accession'].values)
    group_kfold = GroupKFold(n_splits=5)
    
    
    train_cv_score=[]
    cv_score=[]
    train_pearson=[]
    pearson=[]
    train_loss=[]
    loss=[]
    train_delta_tr95=[]
    delta_tr95=[]
    fold_cnt=1
    y_pred=np.asarray([])
    
    for train_idx,test_idx in group_kfold.split(X,y,groups):
        print("Training Fold "+str(fold_cnt))
        model=build_model(max_seq)
        es=EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=30)
        history=model.fit(
            x=X[train_idx],
            y=y[train_idx],
            epochs=TRAIN_EPOCHS,
            batch_size=TRAIN_BATCH_SIZE,
            validation_data=(X[test_idx],y[test_idx]),
            shuffle="batch",
            callbacks=[es]
    #       callbacks=callbacks,
        )
        PlotLosses(history,fold_cnt,output_path)
        #plot_model(model,to_file='./data/RPLC_result/plots/model_prosit.png')
        
        predict_train=model.predict(X[train_idx])[:,0]
            
        pearson_train,r2_train,mse_train,delta_tr95_train=Plot_RT_with_fold(max_rt,min_rt,fold_cnt,y[train_idx],predict_train,output_path+'_train')
        train_cv_score.append(r2_train)
        train_pearson.append(pearson_train)
        train_loss.append(mse_train)
        train_delta_tr95.append(delta_tr95_train)
        
        
        predict_test=model.predict(X[test_idx])[:,0]
        y_pred=np.concatenate((y_pred,predict_test),axis=0)

        pearson_test,r2_test,mse_test,delta_tr95_test=Plot_RT_with_fold(max_rt,min_rt,fold_cnt,y[test_idx],predict_test,output_path+'_test')
        cv_score.append(r2_test)
        pearson.append(pearson_test)
        loss.append(mse_test)
        delta_tr95.append(delta_tr95_test)
        
        model_yaml = model.to_yaml()
        with open(output+"_model.yaml", "w") as yaml_file:
            yaml_file.write(model_yaml)
        # serialize weights to HDF5
        model.save_weights(output+'_fold_'+str(fold_cnt)+"_model_weights.h5")
        print("Saved model to disk.")
        
        fold_cnt+=1
        
    print(train_loss,loss)
    print(train_cv_score,cv_score)
    
    train_cv_score=np.asarray(train_cv_score)
    cv_score=np.asarray(cv_score)
    train_pearson=np.asarray(train_pearson)
    pearson=np.asarray(pearson)
    train_loss=np.asarray(train_loss)
    loss=np.asarray(loss)
    train_delta_tr95=np.asarray(train_delta_tr95)
    delta_tr95=np.asarray(delta_tr95)
    
    print("\n")
    print("%0.6f training accuracy with a standard deviation of %0.6f" % (train_cv_score.mean(), train_cv_score.std())) 
    print("%0.6f accuracy with a standard deviation of %0.6f" % (cv_score.mean(), cv_score.std())) 
    print("%0.6f training Pearson with a standard deviation of %0.6f" % (train_pearson.mean(), train_pearson.std())) 
    print("%0.6f Pearson with a standard deviation of %0.6f" % (pearson.mean(), pearson.std()))
    print("%0.6f train loss with a standard deviation of %0.6f" % (train_loss.mean(), train_loss.std())) 
    print("%0.6f loss with a standard deviation of %0.6f" % (loss.mean(), loss.std()))  
    
    print("%0.6f %0.6f %0.6f %0.6f %0.6f %0.6f" %(train_cv_score.mean(),cv_score.mean(),train_pearson.mean(), pearson.mean(),train_loss.mean(),loss.mean()))
    print("%0.6f %0.6f" %(train_delta_tr95.mean(),delta_tr95.mean()))
    
    df.insert(5,'predict',y_pred)
    loss_test=y_pred
    print(y_pred.shape)
    for i in range(0,y_pred.shape[0]):
      loss_test[i]=y_pred[i]-y[i]
        
    
    df.insert(6,'prediction_loss',loss_test)
 
    df.to_csv(output_path+'_result.tsv',sep='\t',index=None) 
    
    
    
#k_fold_train_with_df(max_seq,max_rt,min_rt,y_attr,data_path,output)   
train_with_df(max_seq,max_rt,y_attr,data_path+"_train.tsv",data_path+"_test.tsv",output)
