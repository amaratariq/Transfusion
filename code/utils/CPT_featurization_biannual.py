#!/usr/bin/env python3

import pandas as pd
import numpy as np
import datetime
import os
import sys
import shutil
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, date


dfcpt_groups = pd.read_csv("code/utils/CPT_group_structure.csv") #cpt code structure
print('No of CPT subgroups:\t', len(dfcpt_groups))
def to_cpt_group(x):
    out=None
    if type(x)==str and x.isnumeric():
        x = int(x)
        temp = dfcpt_groups.loc[(dfcpt_groups['Low']<=x) & (dfcpt_groups['High']>=x) & (dfcpt_groups['Modifier'].isna())]
        if len(temp)>0:
            out = temp.at[temp.index[0], 'Subgroup']
    elif type(x) == str and x[:-1].isnumeric():
        m = x[-1]
        x = int(x[:-1])
        temp = dfcpt_groups.loc[(dfcpt_groups['Low']<=x) & (dfcpt_groups['High']>=x) & (dfcpt_groups['Modifier']==m)]
        if len(temp)>0:
            out = temp.at[temp.index[0], 'Subgroup']
    return out


def procedures_feature_vector_biannual(base_file, cpt_file, out_file):
    """
    base_file: /path/to/file with one hospitalization per row
    cpt_file: /path/to/cpt codes file with one cpt code per row
    out_file: /path/to/output file with one hsopitalization per row along with all CPT subgroups
    """
    ## use this as base file
    df = pd.read_csv(base_file)
    df = df.loc[df.index.repeat(3)] ## 4quarters and day of admission
    df = df.reset_index()
    df['TimePeriod'] = None
    df['admitDate'] = pd.to_datetime(df['ADMIT_DTM'], errors='coerce').dt.date


    df_cpt = pd.read_csv(cpt_file, error_bad_lines=False)  ## all cpt recorded /one cpt code per row
    print('data read', len(df_cpt))
    df_cpt = df_cpt.loc[df_cpt.PATIENT_DK.isin(df.PATIENT_DK)]
    print('relevant data ', len(df_cpt))
    sys.stdout.flush()
    
    
    df_cpt['procedureDate'] = pd.to_datetime(df_cpt['PROCEDURE_DTM'], errors = 'coerce').dt.date
    pd.set_option('mode.chained_assignment', None)
    
    zeros_q1 = 0
    zeros_q2 = 0
    zeros_adm = 0
    for idx in range(0, len(df), 3):
        i = df.index[idx]
        pid = df.at[i, 'PATIENT_DK']
        dt = df.at[i, 'admitDate']  #first 24 hours

        #first quarter
        st = dt-timedelta(days=365)
        ed = st+timedelta(days=182)
        temp = df_cpt.loc[(df_cpt.PATIENT_DK==pid) & (df_cpt.procedureDate>=st) & (df_cpt.procedureDate<=ed)]
        temp2 = temp.drop_duplicates(subset=['PROCEDURE_CODE'])
        d = temp.PROCEDURE_CODE.value_counts()
        if len(temp)>0:
            temp2['SUBGROUPS'] = temp2.PROCEDURE_CODE.apply(to_cpt_group) 
            for ii, jj in temp2.iterrows():
                df.at[i, temp2.at[ii, 'SUBGROUPS']] = d[temp2.at[ii, 'PROCEDURE_CODE']]
        else:
            zeros_q1+=1
        df.at[i, 'TimePerdiod'] = 'Q1' 
        
        #second quarter
        i = df.index[idx+1]
        st = ed+timedelta(days=1) #from next day of the end of last quarter
        ed = dt+timedelta(days=-1)#st+timedelta(days=91)
        temp = df_cpt.loc[(df_cpt.PATIENT_DK==pid) & (df_cpt.procedureDate>=st) & (df_cpt.procedureDate<=ed)]
        temp2 = temp.drop_duplicates(subset=['PROCEDURE_CODE'])
        d = temp.PROCEDURE_CODE.value_counts()
        if len(temp)>0:
            temp2['SUBGROUPS'] = temp2.PROCEDURE_CODE.apply(to_cpt_group) 
            for ii, jj in temp2.iterrows():
                df.at[i, temp2.at[ii, 'SUBGROUPS']] = d[temp2.at[ii, 'PROCEDURE_CODE']]
        else:
            zeros_q2+=1    
        df.at[i, 'TimePerdiod'] = 'Q2'    

           
        #day of admission
        i = df.index[idx+2]
        st = dt
        ed = st+timedelta(days=1)
        temp = df_cpt.loc[(df_cpt.PATIENT_DK==pid) & (df_cpt.procedureDate>=st) & (df_cpt.procedureDate<=ed)]
        temp2 = temp.drop_duplicates(subset=['PROCEDURE_CODE'])
        d = temp.PROCEDURE_CODE.value_counts()
        if len(temp)>0:
            temp2['SUBGROUPS'] = temp2.PROCEDURE_CODE.apply(to_cpt_group) 
            for ii, jj in temp2.iterrows():
                df.at[i, temp2.at[ii, 'SUBGROUPS']] = d[temp2.at[ii, 'PROCEDURE_CODE']]
        else:
            zeros_adm+=1 
        df.at[i, 'TimePerdiod'] = 'D1'    
        print(idx, int(idx/3))      

        sys.stdout.flush()
    print('saving: zeros Q1:{}, Q2:{}, AdmitDay:{}'.format(zeros_q1, zeros_q2, zeros_adm))
    df.to_csv(out_file, index=False)
    