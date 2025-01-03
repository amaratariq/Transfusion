#!/usr/bin/env python3

import pandas as pd
import numpy as np
import pickle as pkl
import os
import sys
import shutil
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, date
import time
print('all imported')
sys.stdout.flush()



def medications_feature_vector_biannual(base_file, med_file, out_file):
    """
    base_file: /path/to/file with one hospitalization per row
    med_file: /path/to/cpt codes file with one cpt code per row
    out_file: /path/to/output file with one hsopitalization per row along with all med subgroups
    """
    ## use this as base file
    df = pd.read_csv(base_file)

    df = df.loc[df.index.repeat(3)] ## 4quarters and day of admission
    df = df.reset_index()
    df['TimePeriod'] = None
    df['admitDate'] = pd.to_datetime(df['ADMIT_DTM'], errors='coerce').dt.date
    

    df_med = pd.read_csv(med_file)
    print(len(df_med), end='\t')
    df_med = df_med.loc[df_med.PATIENT_DK.isin(df.PATIENT_DK.unique())]
    print(len(df_med))
    sys.stdout.flush()
    
    
    df_med['medDate'] = pd.to_datetime(df_med['ADMINISTERED_DTM'], errors = 'coerce').dt.date
    print('patients and medications:\t', len(df),len(df_med))
    sys.stdout.flush()


    pd.set_option('mode.chained_assignment', None)
    
    idx = 0
    jump = 3*jump
    df_med = df_med.loc[df_med.PATIENT_DK.isin(df.iloc[idx:min(idx+jump,len(df))]['PATIENT_DK'])]
    print(idx, len(df_med))
    
    zeros_q1 = 0
    zeros_q2 = 0
    zeros_adm = 0
    for idx in range(0, len(df), 3):
        i = df.index[idx]
        pid = df.at[i, 'PATIENT_DK']
        dt = df.at[i, 'admitDate']   
        
        #first half
        st = dt-timedelta(days=365)
        ed = st+timedelta(days=182)
        temp = df_med.loc[(df_med.PATIENT_DK==pid) & (df_med.medDate>=st)  & (df_med.medDate<=ed) ]   
        if len(temp)>0:
            for c in temp.MED_THERAPEUTIC_CLASS_DESCRIPTION.unique():
                    temp2 = temp.loc[temp.MED_THERAPEUTIC_CLASS_DESCRIPTION==c]
                    df.at[i, c] = len(temp2)          
        else:
            zeros_q1+=1
        df.at[i, 'TimePeriod'] = 'Q1'
        
        #second half
        i = df.index[idx+1]
        st = ed+timedelta(days=1) #from next day of the end of last quarter
        ed = dt+timedelta(days=-1)#1 day before admission
        temp = df_med.loc[(df_med.PATIENT_DK==pid) & (df_med.medDate>=st)  & (df_med.medDate<=ed) ]  
        if len(temp)>0:
            for c in temp.MED_THERAPEUTIC_CLASS_DESCRIPTION.unique():
                    temp2 = temp.loc[temp.MED_THERAPEUTIC_CLASS_DESCRIPTION==c]
                    df.at[i, c] = len(temp2)        
        else:
            zeros_q2+=1
        df.at[i, 'TimePeriod'] = 'Q2'        
        
        #day of admisison
        i = df.index[idx+2]  
        st = dt
        ed = st+timedelta(days=1)
        temp = df_med.loc[(df_med.PATIENT_DK==pid) & (df_med.medDate>=st)  & (df_med.medDate<=ed) ]  
        if len(temp)>0:
            for c in temp.MED_THERAPEUTIC_CLASS_DESCRIPTION.unique():
                    temp2 = temp.loc[temp.MED_THERAPEUTIC_CLASS_DESCRIPTION==c]
                    df.at[i, c] = len(temp2)               
        else:
            zeros_adm+=1
        df.at[i, 'TimePeriod'] = 'D1'
        
        
        print(idx, int(idx/3))

        sys.stdout.flush()
        
        
    print('saving: zeros Q1:{}, Q2:{}, AdmitDay:{}'.format(zeros_q1, zeros_q2, zeros_adm))
    df.to_csv(out_file, index=False)
           
