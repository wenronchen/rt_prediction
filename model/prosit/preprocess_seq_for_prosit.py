#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 16:29:11 2020

@author: wenrchen
"""


##preprocess the sequence for prosit model

import numpy as np
import pandas as pd
import random
#import sys
#data_path=sys.argv[1]
#MAX_SEQUENCE = sys.argv[2]#320,50
#max_rt=2988.05
 ##mcf7: 15759.12 sw480_1:2692.04 yeast:15166.03 sw480_1_0_ptm: 2988.05


ALPHABET = {
    "A": 1,
    "C": 2,
    "D": 3,
    "E": 4,
    "F": 5,
    "G": 6,
    "H": 7,
    "I": 8,
    "K": 9,
    "L": 10,
    "M": 11,
    "N": 12,
    "P": 13,
    "Q": 14,
    "R": 15,
    "S": 16,
    "T": 17,
    "V": 18,
    "W": 19,
    "Y": 20,
}
ALPHABET_S = {integer: char for char, integer in ALPHABET.items()}

def get_sequence_integer(sequences,MAX_SEQUENCE):
    
    array = np.zeros([len(sequences), MAX_SEQUENCE], dtype=int)
    for i, sequence in enumerate(sequences):
        for j, s in enumerate(sequence):
            array[i, j] = ALPHABET[s]
    
    return array 
def get_sequence_integer_with_random(sequences,MAX_SEQUENCE):
    #array = np.zeros([len(sequences), MAX_SEQUENCE], dtype=int)
    array=[]
    for i, seq in enumerate(sequences):
        a=np.zeros([len(seq)])
        for j,s in enumerate(seq):
            a[j]=ALPHABET[s]
        #print(a)
        while(len(a)<MAX_SEQUENCE):
            r=random.randint(0,len(a)-1)
            a=np.insert(a,r,0)
        #print(a)
        array.append(a)
    array=np.asarray(array)
    print(array.shape)
    return array

def per_re(x,max_rt,min_rt):
    x=x/(max_rt-min_rt)
    return x
def normalize_RT(RTs,max_rt,min_rt):
    normalized_rt=RTs
    for i in range(len(RTs)):
        normalized_rt[i]=per_re(normalized_rt[i],max_rt,min_rt)
    return normalized_rt
    

def load_dataset(data_path,MAX_SEQUENCE,max_rt,min_rt,sequence_name,rt_name):
    df=pd.read_csv(data_path,sep='\t')
    sequences=list(df[sequence_name])#bottom-up 'sequence', top-down 'Proteoform'
    print("The length of sequences is ")
    print(len(sequences))
    print(max([len(s) for s in sequences]))
    #x=get_sequence_integer(sequences,MAX_SEQUENCE)
    x=get_sequence_integer(sequences,MAX_SEQUENCE)
    y=list(df[rt_name])#bottom-up 'RT', top-down 'Retention time'
    normalized_y=normalize_RT(y,max_rt,min_rt)
    
    df.insert(2,'normalized_y',normalized_y)
    normalized_y=np.array(normalized_y)
    
    return x, normalized_y, df

    
