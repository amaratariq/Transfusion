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




def filter_vitals(df):
    keys_sel = [
    'Systolic BP (mmHg)',
    'Diastolic BP (mmHg)',
    'Systolic BP',
    'Diastolic Pressure',
    'Systolic Pressure',
    'Non-Invasive SBP', 
    'Non-Invasive DBP', 
    'MAP (mmHg) from Manual Entry',
        'MAP (mmHg)',
        'MAP (mmHg) from Device',
        'MAP',
        'MAP (mmHG) Calculated',

    'SpO2 (%)',
    'SPO2',
    'SpO2',

    'Heart Rate/min',
    'Pulse Rate/min',
    'Pulse',
    'Heart Rate',
    'Pulse rate',
    'Heart rate',
    'Pulse Rate',
    'Pulse Rhythm',
    'Pulse Other',   


    'Temperature (C)',
    'Temperature Value',
    'Skin temperature',
    'Temperature.',
    'Temperature',

    ]

    def str_lower(x):
        return x.lower().strip()

    keys_sel = [k.lower() for k in keys_sel]


    temp = df.loc[df.FLOWSHEET_SUBTYPE_DESCRIPTION.apply(str_lower).isin(keys_sel) 
                    | df.FLOWSHEET_SUBTYPE_DESCRIPTION.apply(str_lower).str.startswith('pain scale')
                    | df.FLOWSHEET_SUBTYPE_DESCRIPTION.apply(str_lower).str.startswith('pain score')
                    | df.FLOWSHEET_SUBTYPE_DESCRIPTION.apply(str_lower).str.startswith('numeric pain')]
    len(temp), len(df), len(temp.PATIENT_DK.unique()), len(df.PATIENT_DK.unique())
    return df

def type_rename(x):
    if 'systolic' in x.lower() or 'sbp' in x.lower():
        return 'SBP'
    if 'diastolic' in x.lower() or 'dbp' in x.lower():
        return 'DBP'
    if 'MAP' in x:
        return 'MAP'
    if 'heart' in x.lower():
        return 'Heart'
    if 'pulse' in x.lower():
        return 'Pulse'
    if 'temp' in x.lower():
        return 'Temperature'
    if 'spo' in x.lower():
        return 'SpO2'
#         if 'pain' in x.lower() and 'scale' in x.lower():
#             return 'Pain'
    if 'pain' in x.lower() and 'score' in x.lower():
        return 'Pain' 

def value_filter(val, subtype):
    if subtype == 'Temperature':
        if val > 80 and val<112: #F
            val =  (val - 32)*5/9 
            return val
        elif val>30 and val<50:
            return val
    if subtype == 'SpO2':
        if val > 50 and val<110: #F
            return val 
    if subtype == 'Pulse' or subtype == 'Heart':
        if val>50 and val<200:
            return val
    if subtype == 'Pain':
        if val>=0 and val<=10:
            return val
    if subtype in  ['MAP', 'SBP', 'DBP']:
        if val>=50 and val<=160:
            return val


def rate_of_change(temp, vital):
    a = [base_values[vital]]+list(temp.FLOWSHEET_RESULT_VAL.values)
    if len(a)>1:
        a_diff = [a[i]-a[i-1] for i in range(1,len(a))]
        return np.mean(a_diff)
    
def rate_of_rate_of_change(temp, vital):
    a = [base_values[vital]]+list(temp.FLOWSHEET_RESULT_VAL.values)
    if len(a)>2:
        a_diff = [a[i]-a[i-1] for i in range(1,len(a))]
        a_diff_diff = [a_diff[i]-a_diff[i-1] for i in range(1,len(a_diff))]
        return np.mean(a_diff_diff)
                                
vitals = ['MAP', 'Pulse', 'Heart', 'SpO2', 'Pain', 'Temperature']
base_values = {'MAP':93.32, 
                'Pulse': 72, 
                'Heart': 72, 
                'SpO2': 100, 
                'Pain': 0,
                'Temperature': 37}

def vitals_feature_vector_daily(base_file_daily, vit_file, out_file):
    """
    base_file_daily: /path/to/file with one day of one hospitalization per row
    vit_file: /path/to/cpt codes file with one vital per row
    out_file: /path/to/output file with one hsopitalization per row along with all selected vitals
    """

    global vitals
    global base_values
    ## use this as base file
    df = pd.read_csv(base_file_daily)
    df['admitDate'] = pd.to_datetime(df['ADMIT_DTM'], errors='coerce').dt.date
    df['Date'] = pd.to_datetime(df['Date']).dt.date

    df_vitals = pd.read_csv(vit_file)
    df_vitals = filter_vitals(df_vitals.copy())
    df_vitals['FLOWSHEET_SUBTYPE_DESCRIPTION'] = df_vitals['FLOWSHEET_SUBTYPE_DESCRIPTION'].apply(type_rename)
    print(df_vitals['FLOWSHEET_SUBTYPE_DESCRIPTION'].value_counts())
    
    

    df_vitals['FLOWSHEET_RESULT_VAL'] = df_vitals.apply(lambda row : value_filter(row['FLOWSHEET_RESULT_VAL'], row['FLOWSHEET_SUBTYPE_DESCRIPTION']), axis = 1)
    df_vitals = df_vitals.dropna(subset=['FLOWSHEET_RESULT_VAL'])
    print(df_vitals['FLOWSHEET_SUBTYPE_DESCRIPTION'].value_counts())
    sys.stdout.flush()

    
    df_vitals = df_vitals.loc[df_vitals.PATIENT_DK.isin(df.PATIENT_DK)]
    df_vit = {}
    for vit in vitals:
        print(vit, end='\t')
        
        if vit == 'MAP':
            temp = df_vitals.loc[(df_vitals.FLOWSHEET_SUBTYPE_DESCRIPTION=='SBP') | (df_vitals.FLOWSHEET_SUBTYPE_DESCRIPTION=='DBP')]
            temp['FLOWSHEET_ASSESSMENT_DTM'] = pd.to_datetime(temp['FLOWSHEET_ASSESSMENT_DTM'])
            temp = temp.sort_values(by='FLOWSHEET_ASSESSMENT_DTM')
            temp_sbp = temp.loc[temp.FLOWSHEET_SUBTYPE_DESCRIPTION.isin(['SBP'])]
            temp_dbp = temp.loc[temp.FLOWSHEET_SUBTYPE_DESCRIPTION.isin(['DBP'])]
            print('SBP:', len(temp_sbp), 'DBP:', len(temp_dbp), end='\t')
            temp_bp = pd.merge_asof(
                                        left = temp_sbp[['FLOWSHEET_RESULT_VAL', 'FLOWSHEET_ASSESSMENT_DTM', 'FLOWSHEET_SUBTYPE_DESCRIPTION', 'PATIENT_DK']], 
                                        right = temp_dbp[['FLOWSHEET_RESULT_VAL', 'FLOWSHEET_ASSESSMENT_DTM', 'FLOWSHEET_SUBTYPE_DESCRIPTION', 'PATIENT_DK']], 
                                        on="FLOWSHEET_ASSESSMENT_DTM", by="PATIENT_DK", tolerance=pd.Timedelta("1000s")
                                    )
            temp_bp = temp_bp.dropna(subset=['FLOWSHEET_RESULT_VAL_x', 'FLOWSHEET_RESULT_VAL_y'], how='any')
            temp_bp = temp_bp.rename(columns={'FLOWSHEET_RESULT_VAL_x': "SBP", 'FLOWSHEET_RESULT_VAL_y': "DBP"})
            def MAP_calculation(sbp, dbp):
                return dbp+(0.333*(sbp-dbp))
            temp_bp['FLOWSHEET_RESULT_VAL'] = temp_bp.apply(lambda row : MAP_calculation(row['SBP'], row['DBP']), axis = 1)
            temp_bp['FLOWSHEET_SUBTYPE_DESCRIPTION'] = 'MAP'
            temp_map =  df_vitals.loc[(df_vitals.FLOWSHEET_SUBTYPE_DESCRIPTION=='MAP')]
            temp = pd.concat([temp_bp, temp_map], ignore_index=True)
            df_vit[vit] = temp.copy()
        else:    
            df_vit[vit] = df_vitals.loc[df_vitals.FLOWSHEET_SUBTYPE_DESCRIPTION==vit]
        df_vit[vit] = df_vit[vit].dropna(subset = ['FLOWSHEET_RESULT_VAL'])
        df_vit[vit]['procedureDate'] = pd.to_datetime(df_vit[vit]['FLOWSHEET_ASSESSMENT_DTM'], errors = 'coerce').dt.date
        df_vit[vit]['procedureDateTime'] = pd.to_datetime(df_vit[vit]['FLOWSHEET_ASSESSMENT_DTM'], errors = 'coerce')
        
        print(len(df_vit[vit]), len(df_vit[vit].PATIENT_DK.unique()))

    pd.set_option('mode.chained_assignment', None)
    
        

    
    for idx in range(0, len(df)):#i,j in df.iterrows():
        i = df.index[idx]
        pid = df.at[i, 'PATIENT_DK']
        dt = df.at[i, 'Date']   

        for v in vitals:
            temp = df_vit[v].loc[(df_vit[v].PATIENT_DK==pid) & (df_vit[v].procedureDate==dt)]

            temp = temp.sort_values(by = 'procedureDateTime')

            df.at[i, v] = rate_of_change(temp, v)
            
        print(idx, i)       
        sys.stdout.flush()

    df.to_csv(out_file, index=False)
    