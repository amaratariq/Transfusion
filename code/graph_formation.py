#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import time
import numpy as np
import sys
import pickle as pkl
import os
import utils 

from sklearn.metrics.pairwise import cosine_similarity as cos_sim
from sklearn.preprocessing import OneHotEncoder as one_enc
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, pairwise_distances
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder

import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.data.utils import load_graphs
from dgl.data.utils import save_graphs
import networkx as nx

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

gcn_msg = fn.copy_src(src='h', out='m')
gcn_mul = fn.u_mul_e('h', 'a', 'm')
gcn_reduce = fn.sum(msg='m', out='h')

import torch
from torch import nn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


header_trans = "path/to/data/"
best_path = "path/to/trained/lstm/"


def strip_string(x):
    if type(x) is str:
        return x.strip()
    else:
        return x


def graph_creation(sim_threshold = 0.99, graph_header = 'graph', node_feats='demo,lab,vit', edge_feats='cpt,icd,med,cpt_icd_med_emb', base_path = 'code/graphs/', **kwargs):
    

    labs = pd.read_csv('selected_labs_expanded_BP.csv')
    labs = list(labs['0'].values)


    icd = ['A00-A09', 'A30-A49', 'B00-B09', 'B15-B19', 'B25-B34', 'B35-B49', 'B95-B97', 'C00-C14', 'C15-C26', 'C30-C39', 'C43-C44', 'C45-C49', 'C50-C50', 'C51-C58', 'C60-C63', 'C64-C68', 'C69-C72', 'C76-C80', 'C81-C96', 'D10-D36', 'D37-D48', 'D49-D49', 'D50-D53', 'D55-D59', 'D60-D64', 'D65-D69', 'D70-D77', 'D80-D89', 'E00-E07', 'E08-E13', 'E15-E16', 'E20-E35', 'E40-E46', 'E50-E64', 'E65-E68', 'E70-E88', 'E89-E89', 'F01-F09', 'F10-F19', 'F30-F39', 'F40-F48', 'F90-F98', 'G20-G26', 'G30-G32', 'G40-G47', 'G50-G59', 'G60-G65', 'G70-G73', 'G80-G83', 'G89-G99', 'H00-H05', 'H25-H28', 'H30-H36', 'H40-H42', 'H49-H52', 'H53-H54', 'H55-H57', 'H60-H62', 'H90-H94', 'I05-I09', 'I10-I16', 'I20-I25', 'I26-I28', 'I30-I52', 'I60-I69', 'I70-I79', 'I80-I89', 'I95-I99', 'J00-J06', 'J09-J18', 'J20-J22', 'J30-J39', 'J40-J47', 'J60-J70', 'J80-J84', 'J90-J94', 'J95-J95', 'J96-J99', 'K00-K14', 'K20-K31', 'K40-K46', 'K50-K52', 'K55-K64', 'K65-K68', 'K70-K77', 'K80-K87', 'K90-K95', 'L00-L08', 'L20-L30', 'L40-L45', 'L49-L54', 'L55-L59', 'L60-L75', 'L76-L76', 'L80-L99', 'M05-M14', 'M15-M19', 'M20-M25', 'M30-M36', 'M40-M43', 'M45-M49', 'M50-M54', 'M60-M63', 'M65-M67', 'M70-M79', 'M80-M85', 'M86-M90', 'N00-N08', 'N10-N16', 'N17-N19', 'N20-N23', 'N25-N29', 'N30-N39', 'N40-N53', 'N60-N65', 'N80-N98', 'Q20-Q28', 'Q60-Q64', 'Q65-Q79', 'T36-T50', 'T80-T88']
    cpt = ['Anesthesia for Procedures on the Upper Abdomen',
     'Surgical Procedures on the Integumentary System',
     'Surgical Procedures on the Musculoskeletal System',
     'Surgical Procedures on the Respiratory System',
     'Surgical Procedures on the Cardiovascular System',
     'Surgical Procedures on the Hemic and Lymphatic Systems',
     'Surgical Procedures on the Digestive System',
     'Surgical Procedures on the Urinary System',
     'Surgical Procedures on the Female Genital System',
     'Surgical Procedures on the Nervous System',
     'Surgical Procedures on the Eye and Ocular Adnexa',
     'Surgical Procedures on the Auditory System',
     'Diagnostic Radiology (Diagnostic Imaging) Procedures',
     'Diagnostic Ultrasound Procedures',
     'Radiologic Guidance',
     'Breast, Mammography',
     'Bone/Joint Studies',
     'Radiation Oncology Treatment',
     'Nuclear Medicine Procedures',
     'Proprietary Laboratory Analyses',
     'Organ or Disease Oriented Panels',
     'Therapeutic Drug Assays',
     'Drug Assay Procedures',
     'Chemistry Procedures',
     'Hematology and Coagulation Procedures',
     'Immunology Procedures',
     'Transfusion Medicine Procedures',
     'Microbiology Procedures',
     'Cytopathology Procedures',
     'Cytogenetic Studies',
     'Surgical Pathology Procedures',
     'Other Pathology and Laboratory Procedures',
     'Immunization Administration for Vaccines/Toxoids',
     'Vaccines, Toxoids',
     'Psychiatry Services and Procedures',
     'Dialysis Services and Procedures',
     'Gastroenterology Procedures',
     'Ophthalmology Services and Procedures',
     'Special Otorhinolaryngologic Services and Procedures',
     'Cardiovascular Procedures',
     'Non-Invasive Vascular Diagnostic Studies',
     'Pulmonary Procedures',
     'Allergy and Clinical Immunology Procedures',
     'Neurology and Neuromuscular Procedures',
     'Hydration, Therapeutic, Prophylactic, Diagnostic Injections and Infusions, and Chemotherapy and Other Highly Complex Drug or Highly Complex Biologic Agent Administration',
     'Physical Medicine and Rehabilitation Evaluations',
     'Medical Nutrition Therapy Procedures',
     'Special Services, Procedures and Reports',
     'Qualifying Circumstances for Anesthesia',
     'Moderate (Conscious) Sedation',
     'Other Medicine Services and Procedures',
     'Medication Therapy Management Services',
     'Non-Face-to-Face Evaluation and Management Services',
     'Transitional Care Evaluation and Management Services',
     'Other Evaluation and Management Services',
     'Pacemaker - Leadless and Pocketless System',
    #  'PR RPR MENINGOCELE <5 CM',
    #  'HC INCISN EXTENSOR TNDN SHTH WRST',
    #  'PR INCISN EXTENSOR TNDN SHTH WRST'
          ]

    meds = ['ANESTHETICS', 'ELECT/CALORIC/H2O', 'ANTICOAGULANTS',
       'ANALGESICS', 'COLONY STIMULATING FACTORS', 'ANTIBIOTICS',
       'GASTROINTESTINAL', 'ANTIVIRALS', 'DIURETICS', 'HORMONES',
       'UNCLASSIFIED DRUG PRODUCTS', 'ANTINEOPLASTICS', 'Unnamed: 142',
       'DIAGNOSTIC', 'ANTIHISTAMINES', 'AUTONOMIC DRUGS', 'SEDATIVE/HYPNOTICS',
       'VITAMINS', 'BLOOD', 'CNS DRUGS', 'ANTIARTHRITICS', 'CARDIOVASCULAR',
       'ANTIPLATELET DRUGS', 'EENT PREPS', 'ANTIDOTES', 'SKIN PREPS',
       'THYROID PREPS', 'PSYCHOTHERAPEUTIC DRUGS', 'ANTIASTHMATICS',
       'MUSCLE RELAXANTS', 'IMMUNOSUPPRESSANTS', 'BIOLOGICALS',
       'ANTIHYPERGLYCEMICS', 'CARDIAC DRUGS', 'ANTIFUNGALS',
       'ANTIPARKINSON DRUGS', 'ANTIINFECTIVES/MISCELLANEOUS',
       'COUGH/COLD PREPARATIONS', 'PRE-NATAL VITAMINS', 'SMOKING DETERRENTS',
       'MISCELLANEOUS MEDICAL SUPPLIES, DEVICES, NON-DRUG', 'CONTRACEPTIVES',
       'ANTI-OBESITY DRUGS', 'ANTIHISTAMINE AND DECONGESTANT COMBINATION',
       'HERBALS', 'ANTIALLERGY',
       'ANTIINFLAM.TUMOR NECROSIS FACTOR INHIBITING AGENTS', 'ANTIPARASITICS',
       'ANALGESIC AND ANTIHISTAMINE COMBINATION']
    

    demo = ['PATIENT_GENDER_NAME', 'PATIENT_RACE_NAME', 'PATIENT_ETHNICITY_NAME', 'PATIENT_AGE_BINNED']
    
    dct = pkl.load(open("encoders.pkl", "rb"))
    enc = dct['demo_enc']
    enc2 = dct['lab_enc']
    keep_idx = dct['demo_keep_idx']
    keep_idx_lab = dct['lab_keep_idx']

    
    graph_name = graph_header+'_threshold_'+str(sim_threshold)
    
    split = pkl.load(open(header_trans+"mrn_split.pkl", "rb"))#header_trans+'mrn_split_3days_or_more_2019.pkl', 'rb'))
    patients_train = split['mrn_train']
    patients_val = split['mrn_val']
    patients_test = split['mrn_test']

    ##load data

    df_cpt = pd.read_csv(header_trans+'cpt_file.csv', low_memory=False)
    df_cpt['TimePeriod'] = df_cpt['TimePerdiod'].copy()
    df_icd = pd.read_csv(header_trans+'icd_file.csv', low_memory=False)
    df_icd['TimePeriod'] = df_icd['TimePerdiod'].copy()
    df_lab = pd.read_csv(header_trans+'lab_file.csv', low_memory=False)
    df_lab['TimePeriod'] = df_lab['TimePerdiod'].copy()
    df_med = pd.read_csv(header_trans+'med_file.csv', low_memory=False)
    
    cols = [c for c in cpt if c not in df_cpt.columns]
    df_cpt[cols] = None
    df_cpt[cols] = df_cpt[cols].fillna(0)

    df_cpt['admitDate'] = pd.to_datetime(df_cpt['admitDate']).dt.date
    df_icd['admitDate'] = pd.to_datetime(df_icd['admitDate']).dt.date
    df_lab['admitDate'] = pd.to_datetime(df_lab['admitDate']).dt.date
    df_med['admitDate'] = pd.to_datetime(df_med['admitDate']).dt.date
    print(len(df_cpt), len(df_icd),  len(df_lab))

    cols = [c for c in cpt if c not in df_cpt.columns]
    df_cpt[cols] = None
    print(len(df_cpt), len(df_icd),  len(df_lab))
    
    
    df = pd.read_csv(header_trans+"raw/base_file.csv")
    df['admitDate'] = pd.to_datetime(df['admitDate']).dt.date
    df = df[[c for c in df.columns if c != "TimePeriod"]].copy()
    print(len(df), len(df_cpt))
    df = df.merge(df_cpt[['PATIENT_DK', 'admitDate', 'TimePeriod']+list(cpt)], on=['PATIENT_DK', 'admitDate'], how='inner') 
    print(len(df))
    df = df.merge(df_icd[['PATIENT_DK', 'admitDate', 'TimePeriod']+list(icd)], on=['PATIENT_DK', 'admitDate', "TimePeriod"], how='left')
    print(len(df))
    
    #RACE Preparation
    races = df.PATIENT_RACE_NAME.unique()
    df['PATIENT_RACE_NAME'] = df['PATIENT_RACE_NAME'].replace({'Unknown':'UNKNOWN', 'Other': 'UNKNOWN', 
                                                        'Choose Not to Disclose': 'UNKNOWN',
                                                        'Unable to Provide': 'UNKNOWN', 
                                                                ' ': 'UNKNOWN'})
    for r in races:
        if type(r) == str and r.startswith('Asian'):
            df['PATIENT_RACE_NAME'] = df['PATIENT_RACE_NAME'].replace({r:'Asian'})
    for r in races:
        if type(r) == str and ('black' in r.lower() or 'african' in r .lower()):
            df['PATIENT_RACE_NAME'] = df['PATIENT_RACE_NAME'].replace({r:'Black'})        
    for r in races:
        if type(r) == str and ('american indian' in r.lower() or 'alaskan' in r .lower()):
            df['PATIENT_RACE_NAME'] = df['PATIENT_RACE_NAME'].replace({r:'American Indian/Alaskan Native'})       
    for r in races:
        if type(r) == str and ('hawaii' in r.lower() or 'pacific' in r .lower() or 'samoan' in r.lower() or 'guam' in r.lower()):
            df['PATIENT_RACE_NAME'] = df['PATIENT_RACE_NAME'].replace({r:'Native Hawaiian/Pacific Islander'})
    df = df.fillna(value={'PATIENT_RACE_NAME':'UNKNOWN'}).copy()
    races = df.PATIENT_ETHNICITY_NAME.unique()
    df['PATIENT_ETHNICITY_NAME'] = df['PATIENT_ETHNICITY_NAME'].replace({'Unknown':'UNKNOWN', 'Other': 'UNKNOWN', 
                                                        'Choose Not to Disclose': 'UNKNOWN',
                                                        'Unable to Provide': 'UNKNOWN'})
    for r in races:
        if type(r) == str and ('cuba' in r.lower() or 'mexic' in r.lower() or 'puerto' in r .lower() or 'central americ' in r.lower() or 'south americ' in r.lower() or 'spanish' in r.lower()):
            df['PATIENT_ETHNICITY_NAME'] = df['PATIENT_ETHNICITY_NAME'].replace({r:'Hispanic or Latino'})
    df = df.fillna(value={'PATIENT_ETHNICITY_NAME':'UNKNOWN'}).copy()       
    def to_bins(x):
        if np.isnan(x):
            return -1
        else:
            if x<100:
                return int(x/10)
            else:
                return 10
    df['PATIENT_AGE_BINNED'] = df['PATIENT_AGE_AT_EVENT'].apply(to_bins)
    print(len(demo), demo)
    print(len(df.columns))
    df = df.loc[:,~df.columns.duplicated()].copy()
    print(len(df.columns))
    sys.stdout.flush()
    df_cpt[cpt] = df_cpt[cpt].fillna(0)
    df_icd[icd] = df_icd[icd].fillna(0)
    df_med[meds] = df_med[meds].fillna(0)
    
    features_icd = icd
    features_cpt = cpt
    sys.stdout.flush()
    
    df_vit = pd.read_csv(header_trans+'vit_file_daily.csv')
    df_vit = df_vit.loc[df_vit.DayNbr==1]
    df_vit = df_vit.drop_duplicates(subset=['PATIENT_DK', 'admitDate'])
    df_vit['TimePeriod'] = 'D1'
    df_vit['admitDate'] = pd.to_datetime(df_vit.admitDate).dt.date
    print(len(df_vit))
    vitals = ['MAP', 'Pulse', 'Heart', 'SpO2', 'Pain', 'Temperature']
    
    

    df = df[['PATIENT_DK', 'admitDate', 'TimePeriod',  "PATIENT_CLINIC_NUMBER", "Transfusion"]+demo].merge(df_cpt[['PATIENT_DK', 'admitDate', 'TimePeriod']+features_cpt], on = ['PATIENT_DK', 'admitDate', 'TimePeriod'], how='left')
    df = df.merge(df_icd[['PATIENT_DK', 'admitDate', 'TimePeriod']+features_icd], on = ['PATIENT_DK', 'admitDate', 'TimePeriod'], how='left')
    df = df.merge(df_lab[['PATIENT_DK', 'admitDate', 'TimePeriod']+labs], on = ['PATIENT_DK', 'admitDate', 'TimePeriod'], how='left')
    df[labs] = df.groupby(['PATIENT_DK', 'admitDate'], sort=False)[labs].ffill()
    df = df.merge(df_vit[['PATIENT_DK', 'admitDate', 'TimePeriod']+vitals], on = ['PATIENT_DK', 'admitDate', 'TimePeriod'], how='left')
    df = df.merge(df_med[['PATIENT_DK', 'admitDate', 'TimePeriod']+meds], on = ['PATIENT_DK', 'admitDate', 'TimePeriod'], how='left')
    
    df_train = df.loc[df.PATIENT_DK.isin(patients_train)]
    df_test = df.loc[df.PATIENT_DK.isin(patients_test)]
    df_val = df.loc[df.PATIENT_DK.isin(patients_val)]

    ### feature matrices
    cols = np.array(cpt)
    x_train_cpt = df_train.loc[df_train.TimePeriod=='D1'][cols].to_numpy().astype('float32')
    x_test_cpt = df_test.loc[df_test.TimePeriod=='D1'][cols].to_numpy().astype('float32')
    x_val_cpt = df_val.loc[df_val.TimePeriod=='D1'][cols].to_numpy().astype('float32')
    
    cols = np.array(icd)
    x_train_icd = df_train.loc[df_train.TimePeriod=='D1'][cols].to_numpy().astype('float32')
    x_test_icd = df_test.loc[df_test.TimePeriod=='D1'][cols].to_numpy().astype('float32')
    x_val_icd = df_val.loc[df_val.TimePeriod=='D1'][cols].to_numpy().astype('float32')
    
    cols = np.array(labs)
    x_train_lab = enc2.transform(df_train.loc[df_train.TimePeriod=='D1'][cols].to_numpy())[:, keep_idx_lab].astype('float32').todense()
    x_test_lab = enc2.transform(df_test.loc[df_test.TimePeriod=='D1'][cols].to_numpy())[:, keep_idx_lab].astype('float32').todense()
    x_val_lab = enc2.transform(df_val.loc[df_val.TimePeriod=='D1'][cols].to_numpy())[:, keep_idx_lab].astype('float32').todense()
    
    x_train_demo = enc.transform(df_train.loc[df_train.TimePeriod=='D1'][demo].to_numpy().reshape(-1,4))[:, keep_idx].astype('float32').todense()
    x_test_demo = enc.transform(df_test.loc[df_test.TimePeriod=='D1'][demo].to_numpy().reshape(-1,4))[:, keep_idx].astype('float32').todense()
    x_val_demo = enc.transform(df_val.loc[df_val.TimePeriod=='D1'][demo].to_numpy().reshape(-1,4))[:, keep_idx].astype('float32').todense()
    
    cols = np.array(vitals)
    x_train_vit = df_train.loc[df_train.TimePeriod=='D1'][cols].fillna(0).to_numpy().astype('float32')
    x_test_vit= df_test.loc[df_test.TimePeriod=='D1'][cols].fillna(0).to_numpy().astype('float32')
    x_val_vit = df_val.loc[df_val.TimePeriod=='D1'][cols].fillna(0).to_numpy().astype('float32')
    
    mm = MinMaxScaler()
    mm.fit(x_train_vit) 
    x_train_vit = mm.transform(x_train_vit) 
    x_test_vit = mm.transform(x_test_vit) 
    x_val_vit = mm.transform(x_val_vit) 
    
    cols = np.array(meds)
    x_train_med = df_train.loc[df_train.TimePeriod=='D1'][cols].to_numpy().astype('float32')
    x_test_med = df_test.loc[df_test.TimePeriod=='D1'][cols].to_numpy().astype('float32')
    x_val_med = df_val.loc[df_val.TimePeriod=='D1'][cols].to_numpy().astype('float32')
      
    
    print('feature vectors created')
    sys.stdout.flush()   
        
    len_train = len(df_train.drop_duplicates(subset=['PATIENT_DK', 'admitDate']))
    len_test = len(df_test.drop_duplicates(subset=['PATIENT_DK', 'admitDate']))
    len_val = len(df_val.drop_duplicates(subset=['PATIENT_DK', 'admitDate']))

    
    train_mask = [True for i in range(len_train)]+[False for i in range(len_val)]+[False for i in range(len_test)]
    val_mask = [False for i in range(len_train)]+[True for i in range(len_val)]+[False for i in range(len_test)]
    test_mask = [False for i in range(len_train)]+[False for i in range(len_val)]+[True for i in range(len_test)]
    
    print('masks created')
    sys.stdout.flush() 
    #labels
    labels_train = df_train.drop_duplicates(subset=['PATIENT_DK', 'admitDate'])['Transfusion'].values
    labels_test = df_test.drop_duplicates(subset=['PATIENT_DK', 'admitDate'])['Transfusion'].values
    labels_val = df_val.drop_duplicates(subset=['PATIENT_DK', 'admitDate'])['Transfusion'].values
    labels = list(labels_train)+list(labels_val)+list(labels_test)+list(labels_ext)
    print('labels created')
    sys.stdout.flush() 
    
    #####Embedding#####

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    class MyLSTM(nn.Module):
        def __init__(self, features, n_hidden=16, num_layers=1):
            super(MyLSTM,self).__init__()
            self.lstm1 = nn.LSTM(input_size=features,hidden_size=n_hidden,num_layers=num_layers,batch_first=True)
            self.lstm2 = nn.LSTM(input_size=n_hidden,hidden_size=features,num_layers=num_layers,batch_first=True)
            self.dropout = nn.Dropout(0.25)
        def forward(self,x):
            output,_ = self.lstm1(x)
            output = self.dropout(output)
            output,_ = self.lstm2(output)
            return output[:,-1,:]
    
    
    ## cpt_icd_med embedding
    temp = df_train.loc[df_train.TimePeriod.isin(['Q1', 'Q2'])][cpt+icd+meds].to_numpy()
    x_train_edge = np.array([temp[i:i+2,:] for i in range(0, temp.shape[0], 2)])
    temp = df_test.loc[df_test.TimePeriod.isin(['Q1', 'Q2'])][cpt+icd+meds].to_numpy()
    x_test_edge = np.array([temp[i:i+2,:] for i in range(0, temp.shape[0], 2)])
    temp = df_val.loc[df_val.TimePeriod.isin(['Q1', 'Q2'])][cpt+icd+meds].to_numpy()
    x_val_edge = np.array([temp[i:i+2,:] for i in range(0, temp.shape[0], 2)])
    
    #replace all zero Q1 with -1
    x_train_edge[np.sum(x_train_edge[:,0,:], axis=1)==0, 0, :] = -1    
    x_test_edge[np.sum(x_test_edge[:,0,:], axis=1)==0, 0, :] = -1
    x_val_edge[np.sum(x_val_edge[:,0,:], axis=1)==0, 0, :] = -1
    
    model = MyLSTM(features = len(cpt)+len(icd)+len(meds), n_hidden=128)
    
    model = utils.load_model_checkpoint(best_path, model) 
    model.eval()
    model.to(device)
    print('embedding model')
    sys.stdout.flush()
    
    X = torch.tensor(x_train_edge, dtype=torch.float).to(device)
    x_train_cpt_icd_med_emb, _ = model.lstm1(X)
    x_train_cpt_icd_med_emb = x_train_cpt_icd_med_emb[:,-1,:].cpu().detach().numpy().squeeze()
    
    X = torch.tensor(x_test_edge, dtype=torch.float).to(device)
    x_test_cpt_icd_med_emb, _ = model.lstm1(X)
    x_test_cpt_icd_med_emb = x_test_cpt_icd_med_emb[:,-1,:].cpu().detach().numpy().squeeze()
    
    X = torch.tensor(x_val_edge, dtype=torch.float).to(device)
    x_val_cpt_icd_med_emb, _ = model.lstm1(X)
    x_val_cpt_icd_med_emb = x_val_cpt_icd_med_emb[:,-1,:].cpu().detach().numpy().squeeze()
    
    
    print('CPT ICD embedding vectors created')
    sys.stdout.flush()
    
    def features_set(n):
        if n=='demo':
            return x_train_demo, x_val_demo, x_test_demo
        elif n=='cpt':
            return x_train_cpt, x_val_cpt, x_test_cpt
        elif n=='icd':
            return x_train_icd, x_val_icd, x_test_icd
        elif n=='lab':
            return x_train_lab, x_val_lab, x_test_lab
        elif n=='vit':
            return x_train_vit, x_val_vit, x_test_vit
        elif n=='med':
            return x_train_med, x_val_med, x_test_med
        elif n=='cpt_icd_med_emb':
            return x_train_cpt_icd_med_emb, x_val_cpt_icd_med_emb, x_test_cpt_icd_med_emb
        else:
            print(n, 'Not Implemented')
            
    node_features = [n.strip() for n in node_feats.split(',')]
    edge_features = [n.strip() for n in edge_feats.split(',')]
    
    n = node_features[0]
    print(n, end='\t')
    print(features_set(n)[0].shape, features_set(n)[1].shape, features_set(n)[2].shape)
    mat_nodes = np.concatenate((features_set(n)[0], features_set(n)[1], features_set(n)[2], features_set(n)[3]), axis=0)
    if len(node_features)>1:
        for n in node_features[1:]:
            print(n, end='\t')
            print(features_set(n)[0].shape, features_set(n)[1].shape, features_set(n)[2].shape)
            mat_nodes = np.concatenate((mat_nodes, 
                                       np.concatenate((features_set(n)[0], features_set(n)[1], features_set(n)[2], features_set(n)[3]), axis=0)
                                       ), axis=1)
    n = edge_features[0]
    print(n, end='\t')
    print(features_set(n)[0].shape, features_set(n)[1].shape, features_set(n)[2].shape)
    mat_edges = np.concatenate((features_set(n)[0], features_set(n)[1], features_set(n)[2], features_set(n)[3]), axis=0)
    if len(edge_features)>1:
        for n in edge_features[1:]:
            print(n, end='\t')
            print(features_set(n)[0].shape, features_set(n)[1].shape, features_set(n)[2].shape)
            mat_edges = np.concatenate((mat_edges, 
                                       np.concatenate((features_set(n)[0], features_set(n)[1], features_set(n)[2], features_set(n)[3]), axis=0)
                                       ), axis=1)
            
            
    print('Characteristics matrix formed', mat_nodes.shape, mat_edges.shape)
    sys.stdout.flush()

    mat_edges[mat_edges>0] = 1
    
    mat_nodes = mat_nodes.astype('uint8')
    mat_edges = mat_edges.astype('uint8')
    
    sim_mat = cos_sim(mat_edges)
                
    sim_mat = np.round(sim_mat, decimals=2)

    print(sim_mat)
    print('Similariy matrix formed')
    sys.stdout.flush()

    graph_name = graph_name+'_'+node_feats+'_'+edge_feats

    ##graph formation
    sim_mat[sim_mat<sim_threshold] = 0
    G = nx.from_numpy_matrix(sim_mat)
    print('graph created')
    sys.stdout.flush()
    sim_arr = sim_mat.flatten()
    edge_weights = sim_arr[sim_arr!=0]
    print(edge_weights.shape)
    print('weights computed')
    sys.stdout.flush()
    g = DGLGraph(G)
    print('DGL graph created')
    sys.stdout.flush()
    g.edata['weights'] = torch.tensor(edge_weights.reshape(-1,1), dtype=torch.float)#torch.from_numpy(edge_weights.reshape(-1,1))
    g.ndata['features'] = torch.tensor(mat_nodes, dtype=torch.float)#torch.from_numpy(mat_nodes)
    
   
    g.ndata['train_mask'] = torch.from_numpy(np.array(train_mask).reshape(-1,1))
    g.ndata['test_mask'] = torch.from_numpy(np.array(test_mask).reshape(-1,1))
    g.ndata['val_mask'] = torch.from_numpy(np.array(val_mask).reshape(-1,1))
    g.ndata['labels'] = torch.from_numpy(np.array(labels).reshape(-1,1))
    
    print('graph weights attached')
    sys.stdout.flush()
    
    
    save_graphs(base_path+graph_name+'.gml', g)
    print(g)
    print('graph saved as\t', graph_name)
    sys.stdout.flush()
    
if __name__ == "__main__":
    graph_creation()
