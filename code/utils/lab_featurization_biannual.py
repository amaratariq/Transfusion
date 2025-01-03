#!/usr/bin/env python3

import pandas as pd
import numpy as np
import datetime
import os
import sys
import shutil
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, date

def map_value(val, val_range):
    try:
        if type(val_range)==str and np.isnan(val)==False:
            if len(val_range.split('-')) == 2:
                lower = float(val_range.split('-')[0])
                upper = float(val_range.split('-')[1])
                if val >= lower and val <= upper:
                    ans = 'NORMAL'
                else:
                    ans = 'ABNORMAL'
            elif '>' in val_range:
                lower = float(''.join(c for c in val_range if (c.isdigit() or c=='.')))
                if val >= lower:
                    ans = 'NORMAL'
                else:
                    ans = 'ABNORMAL'
            elif '<' in val_range:
                upper = float(''.join(c for c in val_range if (c.isdigit() or c=='.')))
                if val <= upper:
                    ans = 'NORMAL'
                else:
                    ans = 'ABNORMAL'
            else:
                ans = 'UNKNOWN'
        elif type(val_range)==str and type(val)==str:
            if val==val_range:
                ans = 'NORMAL'
            else:
                ans = 'ABNORMAL'
        else:
                ans = 'UNKNOWN'
    except:
        ans = 'UNKNOWN'
        print('exception', val, val_range)
    return ans


def labs_feature_vector_biannual(base_file, lab_file, out_file):
    
    """
    base_file: /path/to/file with one hospitalization per row
    lab_file: /path/to/cpt codes file with one lab per row
    out_file: /path/to/output file with one hsopitalization per row along with all selected labs
    """
    ## use this as base file
    df = pd.read_csv(base_file)
    df = df.loc[df.index.repeat(3)] ## 4quarters and day of admission
    df = df.reset_index()
    df['TimePeriod'] = None
    df['admitDate'] = pd.to_datetime(df['ADMIT_DTM'], errors='coerce').dt.date

    
    df_sel_labs = pd.read_csv('selected_labs.csv', header=None)
    sel_labs =  list(df_sel_labs[1].values[1:])
    print(sel_labs)
    
    df_lab = pd.read_csv(lab_file)
    df_lab['LAB_COLLECTION_DTM'] = pd.to_datetime(df_lab['LAB_COLLECTION_DTM'], errors = 'coerce').dt.date
    df_lab - df_lab.loc[df_lab.PATIENT_DK.isin(df.PATIENT_DK.unique())]
    print('patients and conditions:\t', len(df),len(df_lab))
    sys.stdout.flush()


    pd.set_option('mode.chained_assignment', None)

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
        temp = df_lab.loc[(df_lab.PATIENT_DK==pid) & (df_lab.LAB_COLLECTION_DTM>=st) & (df_lab.LAB_COLLECTION_DTM<=ed)]   
        if len(temp)>0:
            for c in temp.LAB_SUBTYPE_CODE.unique():
                temp2 = temp.loc[temp.LAB_SUBTYPE_CODE==c]
                temp2 = temp2.sort_values('LAB_COLLECTION_DTM')
                df.at[i, c] = map_value(temp2.at[temp2.index[-1], 'RESULT_VAL'], temp2.at[temp2.index[-1], 'NORMAL_RANGE_TXT'])           
        else:
            zeros_q1+=1
        df.at[i, 'TimePeriod'] = 'Q1'
        
        #second half
        i = df.index[idx+1]
        st = ed+timedelta(days=1) #from next day of the end of last quarter
        ed = dt+timedelta(days=-1)#1 day before admission
        temp = df_lab.loc[(df_lab.PATIENT_DK==pid) & (df_lab.LAB_COLLECTION_DTM>=st) & (df_lab.LAB_COLLECTION_DTM<=ed)]   
        if len(temp)>0:
            for c in temp.LAB_SUBTYPE_CODE.unique():
                temp2 = temp.loc[temp.LAB_SUBTYPE_CODE==c]
                temp2 = temp2.sort_values('LAB_COLLECTION_DTM')
                df.at[i, c] = map_value(temp2.at[temp2.index[-1], 'RESULT_VAL'], temp2.at[temp2.index[-1], 'NORMAL_RANGE_TXT'])          
        else:
            zeros_q2+=1
        df.at[i, 'TimePeriod'] = 'Q2'
        
        
        #day of admisison
        i = df.index[idx+2]  
        st = dt
        ed = st+timedelta(days=1)
        temp = df_lab.loc[(df_lab.PATIENT_DK==pid) & (df_lab.LAB_COLLECTION_DTM>=st) & (df_lab.LAB_COLLECTION_DTM<=ed)]   
        if len(temp)>0:
            for c in temp.LAB_SUBTYPE_CODE.unique():
                temp2 = temp.loc[temp.LAB_SUBTYPE_CODE==c]
                temp2 = temp2.sort_values('LAB_COLLECTION_DTM')
                df.at[i, c] = map_value(temp2.at[temp2.index[-1], 'RESULT_VAL'], temp2.at[temp2.index[-1], 'NORMAL_RANGE_TXT'])           
        else:
            zeros_adm+=1
        df.at[i, 'TimePeriod'] = 'D12'
        
        
        print(idx, int(idx/3))
        sys.stdout.flush()
        
        
    print('saving : zeros Q1:{}, Q2:{}, AdmitDay:{}'.format(zeros_q1, zeros_q2, zeros_adm))
    df.to_csv(out_file)
           
