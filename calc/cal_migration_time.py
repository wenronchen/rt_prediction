#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: wenrchen
"""

import pandas as pd
#import numpy as np
import sys

## experimental L/((30-2)/L*mobility) 
 

def cal_normalized_mt(mo,L):
    return L*L/(28*mo)

df=pd.read_csv(sys.argv[1],sep='\t')
length=int(sys.argv[3])

min_mo=0.01430172 ##0.01788521 for HeLa ###0.01430172 for SW480 
print(cal_normalized_mt(min_mo,length)) ## used to calculate the maximum normalized migration time
rt=[]
for i in range(df.shape[0]):
    
    mobi=df.iloc[i]['experimental_mobility']
    rt.append(cal_normalized_mt(mobi,length))
    
df.insert(6,'normalized_experimental_mt',rt)
df.to_csv(sys.argv[2],sep='\t',index=None)