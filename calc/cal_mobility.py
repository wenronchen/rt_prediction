#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 11:46:17 2021

@author: wenrchen
"""

##calculate migration mobility 

import pandas as pd
import numpy as np
import sys

def cal_mobility(rt,L):## experimental L/((30-2)/L*tm) (unit of cm2 kV-1 s -1)
    return L*L/(28*rt)

def cal_normalized_rt(mo,L):
    return L*L/(28*mo)

df=pd.read_csv(sys.argv[1],sep='\t')
length=int(sys.argv[3])
#mobi=[]
#for i in range(df.shape[0]):
#    #rt=df.iloc[i]['RT']
#    rt=df.iloc[i]['Feature apex']
#    #rt=np.log2(rt)/np.log2(6000)
#    #rt=df.iloc[i]['Retention time']
#    #rt=df.iloc[i]['Middle retention time']
#    mobi.append(cal_mobility(rt,length))
#    
#df.insert(1,'experimental_mobility',mobi)
#df.to_csv(sys.argv[2],sep='\t',index=None)
#
min_mo=0.01430172 #0.01604478 for ecoli ##0.01788521 for hela ###0.01430172 for SW480 
print(cal_normalized_rt(min_mo,length))
rt=[]
#p_rt=[]
for i in range(df.shape[0]):
    #mobi=2**(df.iloc[i]['experimental_mobility'])
    #p_mobi=2**(df.iloc[i]['normalized_predicted_mt'])
    
    mobi=df.iloc[i]['experimental_mobility']
    #p_mobi=df.iloc[i]['normalized_predicted_mt']
    #rt=df.iloc[i]['Middle retention time']
    rt.append(cal_normalized_rt(mobi,length))
    #p_rt.append(cal_normalized_rt(p_mobi,length))
    
df.insert(6,'normalized_experimental_mt',rt)
#df.insert(7,'renormalized_predicted_mt',p_rt)
df.to_csv(sys.argv[2],sep='\t',index=None)