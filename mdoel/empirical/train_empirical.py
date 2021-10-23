#!/usr/bin python

import sys
import os
import numpy as np
import pandas as pd
import scipy.stats as stats
#import mstp.plot.plot_mobility as plot
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

alphabet='ACDEFGHIKLMNPQRSTVWY'

def getX(df):
  mass=df['Adjusted precursor mass'].values
  charge=df['charge_count'].values
  return [mass, charge]

def predict(mass, charge, b=0.35, c=0.411):
  p_mobi =  np.log(1+b*charge)/mass**c
  return p_mobi


def get_seq_dict(seq,alphabet):
    aa_count=dict( (key, 0) for key in alphabet)
    #print(aa_count)
    
    for s in seq:
        aa_count[s]+=1
    #print(aa_count)
    
    return aa_count



def train_with_empirical(basename,df):

    e_mobi = df['experimental_mobility'].values

    [mass, charge] = getX(df)
    p_mobi = predict(mass, charge)
    
    
    reg = LinearRegression()
    e_mobi = np.transpose(np.matrix(e_mobi))
    reg.fit(e_mobi,p_mobi)
    reg_mobi = reg.predict(e_mobi)
    
    print("R square ",r2_score(reg_mobi,p_mobi))
    df['experimental_mobility'] = reg_mobi
    df.to_csv(basename + "_norm.tsv", sep='\t',index=None)

    loss=abs(reg_mobi-p_mobi)
    print(loss.mean(),np.amax(loss))

    mse = (np.square(reg_mobi - p_mobi)).mean()
    print("mse", mse)
    
    return reg_mobi,p_mobi, reg
# Read file
fname = sys.argv[1]
basename = os.path.splitext(fname)[0]
df=pd.read_csv(fname, sep='\t')
df_unfixed=df.copy()
reg_mobi,p_mobi,reg=train_with_empirical(basename,df_unfixed)

##Deal with the proteoforms with fixed PTMs

fixed_name=sys.argv[2]
basename = os.path.splitext(fixed_name)[0]
df_fixed=pd.read_csv(fixed_name,sep='\t')
e_mobi_fix=df_fixed['experimental_mobility'].values
e_mobi_fix=np.asarray(e_mobi_fix).reshape(-1,1)


[mass_fix,charge_fix]=getX(df_fixed)
p_mobi_fix=predict(mass_fix,charge_fix)

reg_combine=np.concatenate((reg_mobi,reg_mobi_fix),axis=0)
p_combine=np.concatenate((p_mobi,p_mobi_fix[:,0]),axis=0)

print("Combined R square ",r2_score(reg_combine,p_combine))

mse=(np.square(reg_combine-p_combine)).mean()
print("combined mse",mse)

df_fixed.to_csv(basename+'_fixed_norm.tsv',sep='\t',index=None)

df_unfixed=df_unfixed.append(df_fixed,ignore_index=True)
df_unfixed.insert(4,'normalized_predicted_mt',p_combine)
df_unfixed.insert(4,'loss',(reg_combine-p_combine) )
df_unfixed.to_csv(basename+'_with_fixed_norm.tsv',sep='\t',index=None)



