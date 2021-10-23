#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 10:36:47 2021

@author: wenrchen
"""

##calculate charge of proteoforms( count the number of positive charged amino acid at pH2.4)
#count amino acid K,H,R and N-terminal

alphabet='ACDEFGHIKLMNPQRSTVWY'

#import numpy as np
import pandas as pd
import sys

charge_count=[]

df=pd.read_csv(sys.argv[1],sep='\t')

for i in range(df.shape[0]):
    tmp_seq=df.iloc[i]['Proteoform']
    #tmp_seq=df.iloc[i]['sequence']
    tmp_cnt=1## N-terminal count
    for t in tmp_seq:
        if(t=='K' or t== 'H' or t=='R'):
            tmp_cnt+=1
    charge_count.append(tmp_cnt)

df.insert(3,'charge_count',charge_count)
df.to_csv(sys.argv[2],sep='\t',index=None)
