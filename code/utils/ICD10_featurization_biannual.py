#!/usr/bin/env python3

import pandas as pd
import numpy as np
import datetime
import os
import sys
import shutil
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, date
icd = pd.read_csv('code/utils/ICD10_Groups.csv') #ICD10 hierarchy

def find_group(code):
    global icd
    group = ''
    letter = code[0]
    number = code[1:].split('.')[0]
    if number.isnumeric():
        number = (float(number))
        icd_sel = icd.loc[icd.SUBGROUP.str.startswith(letter)].copy()
        icd_sel = icd_sel.loc[(icd_sel.START_IDX.str.isnumeric()) & (icd_sel.END_IDX.str.isnumeric())].copy()
        icd_sel = icd_sel.loc[ (icd_sel.START_IDX.astype(float)<=number) & (icd_sel.END_IDX.astype(float)>=number)].copy()
        if len(icd_sel)>0:
            group = icd_sel.at[icd_sel.index[0], 'SUBGROUP']
        else:
            group = 'UNKNOWN'
    else:
        icd_sel = icd.loc[icd.SUBGROUP.str.startswith(letter)].copy()
        icd_sel = icd_sel.loc[(icd_sel.START_IDX.str.isnumeric()==False) & (icd_sel.END_IDX.str.isnumeric()==False)].copy()
        numheader = number[:-1]
        icd_sel = icd_sel.loc[(icd_sel.START_IDX.str.startswith(numheader)) & (icd_sel.END_IDX.str.startswith(numheader))].copy()
        if len(icd_sel)>0:
            group = icd_sel.at[icd_sel.index[0], 'SUBGROUP']
        else:
            group = 'UNKNOWN'
    return group
    
def conditions_feature_vectors_biannula(base_file, icd_file, out_file):
    """
    base_file: /path/to/file with one hospitalization per row
    icd_file: /path/to/icd codes file with one icd code per row
    out_file: /path/to/output file with one hsopitalization per row along with all ICD10 subgroups
    """
    ## use this as base file
    df = pd.read_csv(base_file)
    
    df_icd = pd.read_csv(icd_file, error_bad_lines=False)  ## all icd recorded /one icd code per row
    print('Length of diagnoses file:\t', len(df_icd), 'No of unique patients in diagnoses file:\t', len(df_icd.PATIENT_DK.unique()))
    sys.stdout.flush()

    df = df.loc[df.index.repeat(3)] ## 2 halves of year and 48 hours of admission
    df = df.reset_index()
    df['TimePeriod'] = None
    df['admitDate'] = pd.to_datetime(df['ADMIT_DTM'], errors='coerce').dt.date
    

    df_icd['diagnosisDate'] = pd.to_datetime(df_icd['DIAGNOSIS_DTM'], errors = 'coerce').dt.date
    df_icd = df_icd.loc[df_icd.PATIENT_DK.isin(df.PATIENT_DK.unique())]
    print('relevant data', len(df_icd))
    sys.stdout.flush()

    pd.set_option('mode.chained_assignment', None)
        

    zeros_q1 = 0
    zeros_q2 = 0
    zeros_adm = 0
    for idx in range(0, len(df),3):
        i = df.index[idx]
        pid = df.at[i, 'PATIENT_DK']
        dt = df.at[i, 'admitDate']   

        #first half
        st = dt-timedelta(days=365)
        ed = st+timedelta(days=182)
        temp = df_icd.loc[(df_icd.PATIENT_DK==pid) & (df_icd.diagnosisDate>=st) & (df_icd.diagnosisDate<=ed)]
        if len(temp)>0:
            temp['SUBGROUPS'] = temp.DIAGNOSIS_CODE.apply(find_group)           
            temp2 = temp.drop_duplicates(subset=['DIAGNOSIS_CODE'])
            d = temp.DIAGNOSIS_CODE.value_counts()
            temp2['SUBGROUPS'] = temp2.DIAGNOSIS_CODE.apply(find_group) 
            for ii, jj in temp2.iterrows():
                df.at[i, temp2.at[ii, 'SUBGROUPS']] = d[temp2.at[ii, 'DIAGNOSIS_CODE']]
        else:
            zeros_q1+=1
        df.at[i, 'TimePerdiod'] = 'Q1'    
        #second half
        i = df.index[idx+1]
        st = ed+timedelta(days=1) #from next day of the end of last quarter
        ed = dt+timedelta(days=-1)#1 day before admission
        temp = df_icd.loc[(df_icd.PATIENT_DK==pid) & (df_icd.diagnosisDate>=st) & (df_icd.diagnosisDate<=ed)]
        if len(temp)>0:
            temp['SUBGROUPS'] = temp.DIAGNOSIS_CODE.apply(find_group)           
            temp2 = temp.drop_duplicates(subset=['DIAGNOSIS_CODE'])
            d = temp.DIAGNOSIS_CODE.value_counts()
            temp2['SUBGROUPS'] = temp2.DIAGNOSIS_CODE.apply(find_group) 
            for ii, jj in temp2.iterrows():
                df.at[i, temp2.at[ii, 'SUBGROUPS']] = d[temp2.at[ii, 'DIAGNOSIS_CODE']]
        else:
            zeros_q2+=1
        df.at[i, 'TimePerdiod'] = 'Q2'      
        #day of admisison
        i = df.index[idx+2]  
        st = dt
        ed = st+timedelta(days=1)
        temp = df_icd.loc[(df_icd.PATIENT_DK==pid) & (df_icd.diagnosisDate>=st) & (df_icd.diagnosisDate<=ed)]
        if len(temp)>0:
            temp['SUBGROUPS'] = temp.DIAGNOSIS_CODE.apply(find_group)           
            temp2 = temp.drop_duplicates(subset=['DIAGNOSIS_CODE'])
            d = temp.DIAGNOSIS_CODE.value_counts()
            temp2['SUBGROUPS'] = temp2.DIAGNOSIS_CODE.apply(find_group) 
            for ii, jj in temp2.iterrows():
                df.at[i, temp2.at[ii, 'SUBGROUPS']] = d[temp2.at[ii, 'DIAGNOSIS_CODE']]
        else:
            zeros_adm+=1
        df.at[i, 'TimePerdiod'] = 'D1'    
        
        print(idx, int(idx/3))      
        sys.stdout.flush()
    print('saving: zeros Q1:{}, Q2:{}, AdmitDay:{}'.format(zeros_q1, zeros_q2, zeros_adm))
    df.to_csv(out_file, index=False)