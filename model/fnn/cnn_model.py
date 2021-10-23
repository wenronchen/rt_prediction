import sys
import time
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self,input_feature,drop_rate,batch_norm,hidden_feature=512):
        super(Net, self).__init__()
        #self.embedding = nn.Embedding(num_embeddings=21, embedding_dim=21,
        #        padding_idx=0)

        # fully connected layer
        if(batch_norm==0):
            self.fc= nn.Sequential(
                    nn.Linear(input_feature,hidden_feature),
                    nn.ReLU(),
                    nn.Linear(hidden_feature,hidden_feature),
                    nn.ReLU(),
                    nn.Dropout(drop_rate),
                    nn.Linear(hidden_feature,hidden_feature),
                    nn.ReLU(),
                    nn.Dropout(drop_rate),
#                    nn.Linear(hidden_feature,hidden_feature),
#                    nn.ReLU(),
#                    nn.Dropout(drop_rate),
#                    nn.Linear(hidden_feature,hidden_feature),
#                    nn.ReLU(),
#                    nn.Dropout(drop_rate),
                    nn.Linear(hidden_feature,1),
                    nn.Sigmoid()
                    )
        else:
             self.fc= nn.Sequential(
                    nn.Linear(input_feature,hidden_feature),
                    nn.BatchNorm1d(hidden_feature),
                    nn.ReLU(),
                    nn.Dropout(drop_rate),
                    nn.Linear(hidden_feature,hidden_feature),
                    nn.BatchNorm1d(hidden_feature),
                    nn.ReLU(),
                    nn.Dropout(drop_rate),
                    nn.Linear(hidden_feature,1),
                    nn.Sigmoid()
                    )
        
        return


    def forward(self, x):

        for layer in self.fc:
            x=layer(x)
        return x

class CNN_plus(nn.Module):
    def __init__(self,input_feature,kernel,out_feature,dilation,drop_rate,pretrained_embedding):
        super(CNN_plus, self).__init__()
        #self.embedding = nn.Embedding(num_embeddings=21, embedding_dim=21,
        #        padding_idx=0)
        if pretrained_embedding is not None:
            self.embed_dim = pretrained_embedding.shape[1]
            self.embedding = nn.Embedding.from_pretrained(pretrained_embedding,
                                                        freeze=False)
            self.embedding_fix=nn.Embedding.from_pretrained(pretrained_embedding,
                                                        freeze=True)
        self.conv1=nn.Conv1d(in_channels=self.embed_dim, #input height
                      out_channels=out_feature, #n_filter
                     kernel_size=kernel, #filter size
                     stride=1,  #filter step
                     dilation=dilation,
                     padding=0)
        self.pool= nn.AvgPool1d(kernel_size=2,stride=1)
        #self.batch_norm=batch_norm
#        if(self.batch_norm!=0):
#            self.bn=nn.BatchNorm1d(out_feature)
        self.flat=nn.Flatten()

            
        # fully connected layer
        self.fc= nn.Sequential(
                nn.Linear((input_feature-(kernel-1)*dilation-1)*out_feature,512),
                nn.ReLU(),
                nn.Dropout(drop_rate),
                nn.Linear(512,512),
                nn.ReLU(),
                nn.Dropout(drop_rate),
                nn.Linear(512,1),
                nn.Sigmoid()
                )
        #self.droput = nn.Dropout(0.5)
        '''
        self.fc= nn.Sequential(
                nn.Linear(2,1),
                )
        '''
        return


    def forward(self, x):
        '''
        x_len=[]
        for i in x:
           x_len.append(torch.count_nonzero(i))
        #print(x_len)
        x_len=torch.Tensor(x_len)
        x_len=torch.reshape(x_len,(x.shape[0],1))

        x_embed = self.embedding(x.long())
        x_embed_fix=self.embedding_fix(x.long())

        x=x_embed.permute(0,2,1)\
        '''
        #x=x.reshape(x.shape[0],x.shape[1],1)
        x_embed = self.embedding_fix(x.long())
        x=x_embed.permute(0,2,1)
        x=self.conv1(nn.functional.relu(x))

        x=self.pool(x)
        x=self.flat(x)
        

        for layer in self.fc:
            x=layer(x)

        return x