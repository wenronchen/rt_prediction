#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 23:00:37 2021

@author: wenrchen
"""

import pandas as pd
import sys
import time

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from matplotlib import pyplot as plt
#from torchsummary import summary
from pytorchtools import EarlyStopping

from sklearn.model_selection import KFold,GroupKFold
from sklearn import metrics

from torch.utils.data import (TensorDataset, DataLoader, RandomSampler, SequentialSampler)
import torch.nn.utils.rnn as rnn_utils
import cnn_model as cnn_model
import hydro_index as hi
import retention_model as rm

if torch.cuda.is_available():       
    device = torch.device("cpu")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

alphabet='ACDEFGHIKLMNPQRSTVWY'

# define a mapping of chars to integers
char_to_int = dict((c, i) for i, c in enumerate(alphabet))
int_to_char = dict((i, c) for i, c in enumerate(alphabet))

#print(one_hot_tensor)


def convert_aa_to_blosum(seq,max_len):
    
    result=[char_to_int[char]+1 for char in seq]
    if(len(result)<max_len):
       while(len(result)!=max_len):
           result.append(0)
    return result

def get_max(df,y_object,evaluation):
    max_mo=0.0
    max_len=0
    for i in range(df.shape[0]):
        if(df.iloc[i][y_object]>max_mo):
            max_mo=df.iloc[i][y_object]
        if(len(df.iloc[i]['Proteoform'])>max_len):
            max_len=len(df.iloc[i]['Proteoform'])
            
    if(evaluation!="none"):
        df_eva=pd.read_csv(evaluation+'.tsv',sep='\t')
        for i in range(df_eva.shape[0]):
            if(df_eva.iloc[i][y_object]>max_mo):
                max_mo=df_eva.iloc[i][y_object]
            if(len(df_eva.iloc[i]['Proteoform'])>max_len):
                max_len=len(df_eva.iloc[i]['Proteoform'])
    print(max_mo,max_len)
    
    return max_mo,max_len

def get_feature_count_input(df,aa):
    seqs=df['Proteoform'].values
    
    seq_feature=[]
    for i in range(seqs.shape[0]):
        cnt=0
        for j in range(len(seqs[i])):
            if(seqs[i][j]==aa):
                cnt+=1
        seq_feature.append(cnt)
    
    seq_feature=np.asarray(seq_feature).reshape(-1,1)
    
    return seq_feature

def get_feature_count_sum(df,aa):
    
    seqs=df['Proteoform'].values
    
    seq_feature=[]
    for i in range(seqs.shape[0]):
        cnt=0
        for j in range(len(seqs[i])):
            for a in aa:
                if(seqs[i][j]==aa):
                    cnt+=1
        seq_feature.append(cnt)
    
    seq_feature=np.asarray(seq_feature).reshape(-1,1)
    
    return seq_feature
    

def get_input_output_with_index(df,y,coding,y_object,max_mo,max_len,feature_size,train_idx,test_idx):


    
    y_train=y[train_idx]
    y_test=y[test_idx]
    
    if(coding=='elude_custom'):
        aaAlphabet=sorted(list(alphabet))
        seqs=np.asarray(df['Proteoform'].values)
        seqs_train=seqs[train_idx]
        
        RT=np.asarray(df[y_object].values)
        RT_train=RT[train_idx]
        
        
        custom_index_train=rm.buildRetentionIndex(aaAlphabet,RT_train,seqs_train,True)
        index=dict(zip(alphabet,custom_index_train))
        print(index)
        
        X=[]

        for seq in seqs:
            feature_vector=rm.computeRetentionFeatureVector(alphabet,seq,index)
            X.append(feature_vector)

        X=np.asarray(X)
        if(feature_size!=0):
            feature_index=[i for i in range(feature_size)]
            X=X[:,feature_index]
            
        X_train=X[train_idx]
        X_test=X[test_idx]   
            
    return X_train,X_test,y_train,y_test
    

def get_input_output(df,coding,y_object,max_mo,max_len,feature_size):
    X=[]
    y=[]
    embedding_tensor=0
    df=df.copy()
    y=np.asarray(df[y_object].values)
    y=y/max_mo
    
    if(coding=='one-hot'):
        for i in range(0,df.shape[0]):
           tmp=df.iloc[i]['Proteoform']
           tmp_int=[char_to_int[char] for char in tmp]
           X.append(torch.Tensor(tmp_int))
           
        X=rnn_utils.pad_sequence(X,batch_first=True)
        
        one_hot_tensor=torch.eye(21)
        for i in range(21):
            one_hot_tensor[i][i]=hi.hydro_index_a[i]
        #one_hot_tensor=np.c_[one_hot_tensor,torch.Tensor(np.reshape(hi.hydro_index_a,(21,1)))]
        #print(one_hot_tensor)
        
        embedding_tensor=one_hot_tensor
    elif(coding=='elude_custom'):
        
        aaAlphabet=sorted(list(alphabet))
        seqs=np.asarray(df['Proteoform'].values)
        RT=np.asarray(df[y_object].values)
        
        custom_index=rm.buildRetentionIndex(aaAlphabet,RT,seqs,True)
        index=dict(zip(alphabet,custom_index))
        
        X=[]

        for seq in seqs:
            feature_vector=rm.computeRetentionFeatureVector(alphabet,seq,index)
            X.append(feature_vector)

        X=np.asarray(X)
        if(feature_size!=0):
            feature_index=[i for i in range(feature_size)]
            X=X[:,feature_index]

        
        
        
    else:
        print(coding)
        if(coding.find('mz')!=-1):
            Z=np.asarray(df['charge_count'].values).reshape(-1,1)
            seqs_m=(df['Adjusted precursor mass'].values)
            seqs_m=np.asarray(seqs_m).reshape(-1,1)
            X=np.concatenate((Z/20,seqs_m/20000),axis=1)
        if(coding.find('mh')!=-1):
            seqs_m=(df['Adjusted precursor mass'].values)
            seqs_m=np.asarray(seqs_m).reshape(-1,1)
            HI=np.asarray(df['hydro_index_sum'].values).reshape(-1,1)
            X=np.concatenate((HI/2000,seqs_m/20000),axis=1)
            
        if(coding.find('fixed')!=-1):
            ptm_cnt=[]
            for i in range(df.shape[0]):
                if(df.iloc[i]['fixed_PTM']==1):
                    ptm_cnt.append(df.iloc[i]['Proteoform'].count('C'))
                else:
                    ptm_cnt.append(0)
            ptm_cnt=np.asarray(ptm_cnt).reshape(-1,1)
            X=np.concatenate((X,ptm_cnt/20),axis=1)

        if(coding.find('+')!=-1):
            flag=coding.find('+')
            coding=coding[flag+1:]
            if(coding.find('(')==-1):
                for f in coding:
                    tmp=get_feature_count_input(df,f)
                #print(tmp.shape)
                    X=np.concatenate((X,tmp/20),axis=1)
            else:
                while(coding.find('(')!=-1):
                    flag1=coding.find('(')
                    flag2=coding.find(')')
                    
                    f=coding[flag1+1:flag2]
                    tmp=get_feature_count_sum(df,f)
                    X=np.concatenate((X,tmp/20),axis=1)
                    
                    coding=coding[flag2+1:]
    
    
    return X,y,embedding_tensor

def data_loader(train_x, test_x, train_y, test_y,
                batch_size=512):
    """Convert train and test sets to torch.Tensors and load them to
    DataLoader.
    """
    
    train_x=torch.Tensor(train_x)
    train_y=torch.Tensor(train_y)
    
     # Create DataLoader for training data
    train_dataset = TensorDataset(train_x, train_y)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
        
    
    test_x=torch.Tensor(test_x)
    test_y=torch.Tensor(test_y)
    
    # Create DataLoader for testidation data
    test_dataset = TensorDataset(test_x, test_y)

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    return train_dataloader, test_dataloader


def PlotLosses(logs,output,fold):
  plt.subplot()  
  plt.plot(logs['train_loss'])   
  plt.plot(logs['test_loss'])
  plt.title('model loss')  
  plt.ylabel('loss')  
  plt.xlabel('epoch')  
  plt.ylim([0,0.03])
  plt.legend(['train','test'], loc='upper right')
  plt.savefig(output+'_fold'+str(fold)+'_loss.png')
  #plt.show()
  plt.close()
  




def train(model, output,fold,optimizer,loss_fn,x_train,x_test,y_train,y_test,batch_size,epochs):
    """Train the CNN model."""
    logs={}
    logs['train_loss']=[]
    #logs['val_loss']=[]
    logs['test_loss']=[]
    # Tracking best validation accuracy
    best_loss = 100
    
    train_dataloader, test_dataloader = data_loader(x_train, x_test, y_train, y_test, batch_size)
    
    early_stopping = EarlyStopping(patience=100, verbose=True,path=output+'_fold_'+str(fold)+'_checkpoint.pt')

    # Start training loop
    print("Start training...\n")
    print(f"{'Epoch':^7} | {'Train Loss':^12} |  {'Test Loss':^10} | {'Elapsed':^9}")
    print("-"*60)
    train_output=[]
    #val_output=[]
    for epoch_i in range(epochs):
        # =======================================
        #               Training
        # =======================================
        # Tracking time and loss
        t0_epoch = time.time()
        ##tracking the loss for training and validation
        total_loss = 0
        train_output=[]
        #val_output=[]
        # Put the model into the training mode
        model.train()

        for step, batch in enumerate(train_dataloader):
            # Load batch to GPU
            b_input_ids, b_labels = tuple(t.to(device) for t in batch)
            b_labels=b_labels.reshape(b_labels.shape[0],1)
            # Zero out any previously calculated gradients
            optimizer.zero_grad()

            # Perform a forward pass. This will return logits
            
            logits = model(b_input_ids)
            #
            if torch.cuda.is_available():
                logits_numpy=logits.detach().cpu().numpy()
            else:
                logits_numpy=logits.detach().numpy()
            train_output.append(logits_numpy)
            # Compute loss and accumulate the loss values
            loss = loss_fn(logits, b_labels)
            total_loss += loss.item()

            # Perform a backward pass to calculate gradients
            loss.backward()

            # Update parameters
            optimizer.step()

        # Calculate the average loss over the entire training data
        avg_train_loss = total_loss / len(train_dataloader)
        logs['train_loss'].append(avg_train_loss)
        
        
        
        # =======================================
        #               Evaluation
        # =======================================
        if test_dataloader is not None:
            # After the completion of each training epoch, measure the model's
            # performance on our testidation set.
            test_loss, test_output= evaluate(model, loss_fn,test_dataloader)
            logs['test_loss'].append(test_loss)
            # Track the best accuracy
            if test_loss < best_loss:
                best_loss = test_loss

            # Print performance over the entire training data
            time_elapsed = time.time() - t0_epoch
            print(f"{epoch_i + 1:^7} | {avg_train_loss:^12.6f} |  {test_loss:^10.6f} | {time_elapsed:^9.2f}")
            
            early_stopping(test_loss, model)
        
            if early_stopping.early_stop:
                print("Early stopping")
                break
            
    PlotLosses(logs,output,fold)
    
    print("\n")
    print(f"Training complete! Best test loss: {best_loss:.4f}.")
    
    
    
    #train_output=train_output+val_output
    
    train_output=[a.squeeze().tolist() for a in train_output]
    test_output=[a.squeeze().tolist() for a in test_output]
    
    #train_flat=np.asarray([item for sublist in train_output for item in sublist])
    train_flat=np.asarray(list(pd.core.common.flatten(train_output)))
    test_flat=np.asarray(list(pd.core.common.flatten(test_output)))
    #print(test_flat.shape)
    
    
    return train_flat,test_flat,logs['train_loss'][-1],logs['test_loss'][-1]

def evaluate(model,loss_fn, val_dataloader):
    """After the completion of each training epoch, measure the model's
    performance on our validation set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled
    # during the test time.
    model.eval()

    # Tracking variables
    #val_accuracy = []
    val_loss = []
    val_output= []

    for batch in val_dataloader:
        # Load batch to GPU
        b_input_ids, b_labels = tuple(t.to(device) for t in batch)
        b_labels=b_labels.reshape(b_labels.shape[0],1)
        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids.float())

        # Compute loss
        loss = loss_fn(logits, b_labels)
        val_loss.append(loss.item())
        
        if torch.cuda.is_available():
            logits_numpy=logits.detach().cpu().numpy()
        else:
            logits_numpy=logits.detach().numpy()
        val_output.append(logits_numpy)


    # Compute the average accuracy and loss over the validation set.
    val_loss = np.mean(val_loss)

    return val_loss,val_output

def evalution_with_other_data(evalution,df,model,output,loss_fn,coding,y_object,batch_size,fold_cnt,max_rt):
    df_eva=pd.read_csv(evalution+'.tsv', sep='\t')
    print(df_eva.shape)
    #max_mo,max_len=get_max(df,y_object,evaluation)

    X_eva,y_eva,_=get_input_output(df_eva,coding,y_object,max_rt,max_len)
    X_eva_tensor=torch.tensor(X_eva)
    y_eva_tensor=torch.Tensor(y_eva)
    
    eva_dataset = TensorDataset(X_eva_tensor, y_eva_tensor)
    eva_sampler = SequentialSampler(eva_dataset)
    eva_dataloader = DataLoader(eva_dataset, sampler=eva_sampler, batch_size=batch_size)
    
    eva_loss,eva_output=evaluate(model,loss_fn,eva_dataloader)
    eva_output=[a.squeeze().tolist() for a in eva_output]
    eva_flat=np.asarray([item for sublist in eva_output for item in sublist])
    
    eva_score=metrics.r2_score(y_eva,eva_flat)
    
    df_eva.insert(2,'predict',eva_flat)
    loss_eva=eva_flat
    for i in range(0,eva_flat.shape[0]):
        loss_eva[i]=eva_flat[i]-y_eva[i]
    df_eva.insert(3,'loss',loss_eva)
    df_eva.to_csv(output+'_evaluation_'+str(fold_cnt)+'fold_result.tsv',sep='\t',index=None)
    
    return eva_loss,eva_score

def one_fold_train(evaluation,df_name,coding,y_object,feature_size,embedding_tensor,output,max_len,drop_rate,batch_norm,activation,hidden_feature,X,y,k,epoch,batch_size,lr,max_rt):
    
    train_df=pd.read_csv(df_name+'_train.tsv',sep='\t')
    
    test_df=pd.read_csv(df_name+'_test.tsv',sep='\t')
    
    df_combine=train_df.append(test_df,ignore_index=True)
    train_idx=[idx for idx in range(train_df.shape[0])]
    test_idx=[idx for idx in range(train_df.shape[0],df_combine.shape[0])]
    
    y=np.asarray(df_combine[y_object].values)
    y=y/max_rt
    

    model=cnn_model.Net(max_len,drop_rate,batch_norm,hidden_feature)
        
    print(model)
    model.to(device)
    
    optimizer = Adam(model.parameters(),lr=lr)
    loss_fn = nn.MSELoss()
    
    train_cv_score=[]
    cv_score=[]
    train_loss=[]
    loss=[]
    fold_cnt=1
    y_pred=np.asarray([])
    
    if(coding=="elude_custom"):
        X_train,X_test,y_train,y_test=get_input_output_with_index(df_combine,y,coding,y_object,max_mo,max_len,feature_size,train_idx,test_idx)
        train_output,test_output,train_loss_tmp,test_loss_tmp=\
        train(model,output,fold_cnt,optimizer,loss_fn,X_train,X_test,y_train,y_test,batch_size,epoch)
        
        train_cv_score.append(metrics.r2_score(y_train,train_output))
        cv_score.append(metrics.r2_score(y_test,test_output))
    else:
        X,y,embedding_tensor=get_input_output(df_combine,coding,y_object,max_rt,max_len,feature_size)
        
        train_output,test_output,train_loss_tmp,test_loss_tmp=\
        train(model,output,fold_cnt,optimizer,loss_fn,X[train_idx],X[test_idx],y[train_idx],y[test_idx],batch_size,epoch)
    
        train_cv_score.append(metrics.r2_score(y[train_idx],train_output))
        cv_score.append(metrics.r2_score(y[test_idx],test_output))
    y_pred=np.concatenate((y_pred,test_output),axis=0)

    train_loss.append(train_loss_tmp)
    loss.append(test_loss_tmp)
        
    print(train_loss,loss)
    print(train_cv_score,cv_score)
    print(train_cv_score,'\t',cv_score,'\t',train_loss,'\t',loss)
    
    train_cv_score=np.asarray(train_cv_score)
    cv_score=np.asarray(cv_score)
    train_loss=np.asarray(train_loss)
    loss=np.asarray(loss)
    
    print("\n")
    print("%0.6f training accuracy with a standard deviation of %0.6f" % (train_cv_score.mean(), train_cv_score.std())) 
    print("%0.6f accuracy with a standard deviation of %0.6f" % (cv_score.mean(), cv_score.std())) 
    print("%0.6f train loss with a standard deviation of %0.6f" % (train_loss.mean(), train_loss.std())) 
    print("%0.6f loss with a standard deviation of %0.6f" % (loss.mean(), loss.std()))  
    
    return y_pred
    
            
    
def k_fold_train(evaluation,df,coding,y_object,feature_size,embedding_tensor,output,max_len,drop_rate,batch_norm,activation,hidden_feature,X,y,k,epoch,batch_size,lr,max_rt):
    
    groups=np.asarray(df['Protein accession'].values)
    k_fold=GroupKFold(n_splits=k)
    
    train_cv_score=[]
    cv_score=[]
    train_loss=[]
    loss=[]
    fold_cnt=1
    y_pred=np.asarray([])
    if(evaluation!='none'):
        eva_loss=[]
        eva_score=[]
    
    for train_idx,test_idx in k_fold.split(X,y,groups):
        
        
        model=cnn_model.Net(max_len,drop_rate,batch_norm,hidden_feature)
            
        print(model)
        model.to(device)
        
        optimizer = Adam(model.parameters(),lr=lr)
        loss_fn = nn.MSELoss()
        
        if(coding=="elude_custom"):
            X_train,X_test,y_train,y_test=get_input_output_with_index(df,y,coding,y_object,max_rt,max_len,feature_size,train_idx,test_idx)
            train_output,test_output,train_loss_tmp,test_loss_tmp=\
            train(model,output,fold_cnt,optimizer,loss_fn,X_train,X_test,y_train,y_test,batch_size,epoch)
            
            train_cv_score.append(metrics.r2_score(y_train,train_output))
            cv_score.append(metrics.r2_score(y_test,test_output))
        else:
            
            
            train_output,test_output,train_loss_tmp,test_loss_tmp=\
            train(model,output,fold_cnt,optimizer,loss_fn,X[train_idx],X[test_idx],y[train_idx],y[test_idx],batch_size,epoch)
        
            train_cv_score.append(metrics.r2_score(y[train_idx],train_output))
            cv_score.append(metrics.r2_score(y[test_idx],test_output))
        y_pred=np.concatenate((y_pred,test_output),axis=0)

        train_loss.append(train_loss_tmp)
        loss.append(test_loss_tmp)

        
        if(evaluation!='none'):
            tmp_eva_loss,tmp_eva_score=evalution_with_other_data(evaluation,df,model,output,loss_fn,coding,y_object,batch_size,fold_cnt)
            eva_loss.append(tmp_eva_loss)
            eva_score.append(tmp_eva_score)
        
        
        fold_cnt+=1
    print(train_loss,loss)
    print(train_cv_score,cv_score)
    
    train_cv_score=np.asarray(train_cv_score)
    cv_score=np.asarray(cv_score)
    train_loss=np.asarray(train_loss)
    loss=np.asarray(loss)
    
    if(evaluation!='none'):
        print(eva_loss,eva_score)
        eva_loss=np.asarray(eva_loss)
        eva_score=np.asarray(eva_score)
        print("%0.6f evaluation accuracy with a standard deviation of %0.4f" % (eva_score.mean(), eva_score.std()))
        print("%0.6f evaluation loss with a standard deviation of %0.4f" % (eva_loss.mean(), eva_loss.std())) 
    
    print("\n")
    print("%0.6f training accuracy with a standard deviation of %0.6f" % (train_cv_score.mean(), train_cv_score.std())) 
    print("%0.6f accuracy with a standard deviation of %0.6f" % (cv_score.mean(), cv_score.std())) 
    print("%0.6f train loss with a standard deviation of %0.6f" % (train_loss.mean(), train_loss.std())) 
    print("%0.6f loss with a standard deviation of %0.6f" % (loss.mean(), loss.std()))  
    
    print("%0.6f %0.6f %0.6f %0.6f" %(train_cv_score.mean(),cv_score.mean(),train_loss.mean(),loss.mean()))

    
    return y_pred


input_path=sys.argv[1]
output_path=sys.argv[2]
coding=sys.argv[3]##The method for coding sequences:One-hot encoding and Blosum62 encoding or count of features

k=int(sys.argv[4])
batch_size=int(sys.argv[5])
EPOCH=int(sys.argv[6])
activation=sys.argv[7]
hidden_feature=int(sys.argv[8])
y_object=sys.argv[9]
lr=float(sys.argv[10])## The learning rate of optimizer,default=0.0001
drop_rate=float(sys.argv[11])
feature_size=int(sys.argv[12])
batch_norm=0
evaluation="none"



df=pd.read_csv(input_path+'_test.tsv',sep='\t')
#df=pd.read_csv(input_path+'.tsv',sep='\t')

max_mo,max_len=get_max(df,y_object,evaluation)
max_rt=float(sys.argv[13])
X,y,embedding_tensor=get_input_output(df,coding,y_object,max_mo,max_len,feature_size)



if(coding !='one-hot'):
    if(coding.find('elude')!=-1):
        input_feature=X.shape[1]
    else:   
        if(coding.find('fixed')!=-1):
            flag=coding.find('fixed')
            coding_modified=coding[:flag+1]+coding[flag+5:]
        else:
            coding_modified=coding
        if(coding.find('+')==-1):
            input_feature=len(coding_modified)
        else:
            if(coding.find('(')==-1):
                input_feature=len(coding_modified)-1
            else:
                flag=coding_modified.find('(')
                coding1=coding_modified[:flag-1]
                coding2=coding_modified[flag:]
                input_feature=len(coding1)+coding2.count('(')
    max_len=input_feature
print("Input feature length=", max_len)

#y_pred=k_fold_train(evaluation,df,coding,y_object,feature_size,embedding_tensor,output_path,max_len,drop_rate,batch_norm,activation,hidden_feature,X,y,k,EPOCH,batch_size,lr,max_rt)

if(k==1):
	y_pred=one_fold_train(evaluation,input_path,coding,y_object,feature_size,embedding_tensor,output_path,max_len,drop_rate,batch_norm,activation,hidden_feature,X,y,k,EPOCH,batch_size,lr,max_rt)

#plt.scatter(y*max_rt, y_pred*max_rt,  color='red')
#lims=np.linspace(0,max_rt,1000).reshape(-1,1)
#plt.plot(lims,lims,linestyle='--',color='blue',label='reference')
#plt.xlabel('experimental')
#plt.ylabel('predicted')
#plt.legend()
#plt.savefig(output_path+'_'+str(k)+'fold_cnn_'+str(coding)+'result_plot'+'.png')
#plt.show()
#plt.close()
        
        
df.insert(5,'predicted',y_pred)
loss_test=y_pred
for i in range(0,y_pred.shape[0]):
    loss_test[i]=y_pred[i]-y[i]
        
    
df.insert(6,'prediction_loss',loss_test) 
df.to_csv(output_path+'_'+str(k)+'fold_'+str(coding)+'_result.tsv',sep='\t',index=None)   


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
