import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import math
import pickle as pkl
import pickle
import numpy as np
from collections import Counter
import sys
import random
import json
import utils

from sklearn.metrics import mean_squared_error, pairwise_distances, classification_report
from sklearn.metrics.pairwise import cosine_similarity as cos_sim
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, pairwise_distances, classification_report
from sklearn.decomposition import PCA

import torch
from torch import nn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


## set random seed
rand_seed = 0
utils.seed_torch(seed=rand_seed)

header_data = 'path/to/data/' #data/
header_out  = 'path/to/output/' #code/models/lstm/


## features

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


## custom dataset
class sequence(Dataset):
    def __init__(self,df, df_cpt, df_icd, df_med, cpt, icd,  meds, seq_len):
        self.df = df
        self.len = len(self.df)
        self.icd = icd
        self.cpt = cpt
        self.meds = meds
        self.seq_length = seq_len
        
        self.df_cpt = df_cpt
        self.df_icd = df_icd
        self.df_med = df_med
        
    def __getitem__(self,idx):
        pid = self.df.loc[self.df.index[idx], 'PATIENT_DK']
        dt = self.df.loc[self.df.index[idx], 'admitDate']
        X_cpt = self.df_cpt.loc[(self.df_cpt.PATIENT_DK==pid) & (self.df_cpt.admitDate==dt)][self.cpt].to_numpy().astype('float32')
        X_icd = self.df_icd.loc[(self.df_icd.PATIENT_DK==pid) & (self.df_icd.admitDate==dt)][self.icd].to_numpy().astype('float32')
        X_med = self.df_med.loc[(self.df_med.PATIENT_DK==pid) & (self.df_med.admitDate==dt)][self.meds].to_numpy().astype('float32')
        
        X = np.concatenate((X_cpt, X_icd, X_med), axis=1)
        if np.sum(X[0,:])==0:
            X[0,:]=-1
        X[X>0] = 1        
        X = torch.tensor(X, dtype=torch.float)        
        
        Y = X[2,:]
        X = X[:2,:]
        
        return X,Y
    def __len__(self):
        return self.len

## custom model
class MyLSTM(nn.Module):
    def __init__(self, features, n_hidden, num_layers):
        super(MyLSTM,self).__init__()
        self.lstm1 = nn.LSTM(input_size=features,hidden_size=n_hidden,num_layers=num_layers,batch_first=True)
        self.lstm2 = nn.LSTM(input_size=n_hidden,hidden_size=features,num_layers=num_layers,batch_first=True)
        self.dropout = nn.Dropout(0.25)
        

    def forward(self,x):
        output,_ = self.lstm1(x)
        output = self.dropout(output)
        output,_ = self.lstm2(output)
        return output[:,-1,:]

def encoder_model_development():
    global header_data
    global header_out
    ## load data
    split = pkl.load(open(header_data+"mrn_split.pkl", "rb"))
    patients_train = split['mrn_train']
    patients_val = split['mrn_val']
    patients_test = split['mrn_test']

    features_icd = icd
    features_cpt = cpt

    df = pd.read_csv(header_data+'raw/base_file.csv', low_memory=False)
    df_cpt = pd.read_csv(header_data+'cpt_file.csv', low_memory=False) #3 intervals per patient
    cols = [c for c in cpt if c not in df_cpt.columns]
    df_cpt[cols] = None
    df_icd = pd.read_csv(header_data+'icd_file.csv', low_memory=False) #3 intervals per patient
    df_med = pd.read_csv(header_data+'med_file.csv', low_memory=False) #3 intervals per patient
    df_cpt['admitDate'] = pd.to_datetime(df_cpt['admitDate']).dt.date
    df_icd['admitDate'] = pd.to_datetime(df_icd['admitDate']).dt.date
    df_med['admitDate'] = pd.to_datetime(df_med['admitDate']).dt.date
    df_cpt['TimePeriod'] = df_cpt['TimePerdiod'].copy()
    df_icd['TimePeriod'] = df_icd['TimePerdiod'].copy()

    df_cpt[features_cpt] = df_cpt[features_cpt].fillna(0)
    df_icd[features_icd] = df_icd[features_icd].fillna(0)
    df_med[meds] = df_med[meds].fillna(0)

    df_train = df.loc[df.PATIENT_CLINIC_NUMBER.isin(patients_train)]
    df_val = df.loc[df.PATIENT_CLINIC_NUMBER.isin(patients_val)]
    df_test = df.loc[df.PATIENT_CLINIC_NUMBER.isin(patients_test)]

    df_cpt[features_cpt] = df_cpt[features_cpt].fillna(0)
    df_icd[features_icd] = df_icd[features_icd].fillna(0)
    df_med[meds] = df_med[meds].fillna(0)


    ## training

    do_train=True
    best_path = None
    n_iters = 50#
    print_every = 1
    plot_every = 1
    batch_size = 512
    seq_len = 7
    features = len(features_cpt)+len(features_icd)+len(meds)
    n_hidden = 128
    num_layers = 1
    lr = 1e-4
    model_name = 'lstm'
    save_dir = header_out
    metric_name = 'auroc'
    maximize_metric=True

    dataset_train = sequence(df_train, df_cpt, df_icd, df_med,  cpt, icd,  meds,  seq_len)
    train_loader = DataLoader(dataset_train,shuffle=True,batch_size=batch_size)
    dataset_test = sequence(df_test, df_cpt, df_icd, df_med,   cpt, icd, meds, seq_len)
    test_loader = DataLoader(dataset_test,shuffle=True,batch_size=batch_size)
    dataset_val = sequence(df_val, df_cpt, df_icd, df_med,   cpt, icd, meds,  seq_len)
    val_loader = DataLoader(dataset_val,shuffle=True,batch_size=batch_size)

    model = MyLSTM(features, n_hidden, num_layers)


    save_dir = utils.get_save_dir(
        save_dir, training=True if do_train else False
    )

    logger = utils.get_logger(save_dir, "LSTM")
    logger.propagate = False

    logger.info('do_train: {}, n_iter: {}, batch_size: {}, seq_len: {}, features: {}, hidden dimensions: {}, num_layers: {}, reading from: {}'.format(do_train, n_iters, batch_size, seq_len, features, n_hidden, num_layers, best_path))
    logger.info('Model {}'.format(model))
    sys.stdout.flush()
    ## checkpoint saver
    saver = utils.CheckpointSaver(
        save_dir=save_dir,
        metric_name=metric_name,
        maximize_metric=maximize_metric,
        log=logger,
    )

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)

    current_loss = 0
    all_losses = []

    cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info('cuda {}, Device {}'.format(cuda, device))
    logger.info('saving to {}'.format(save_dir))
    if cuda:
        model = model.to(device)
        model.train()

    train_losses = []
    val_losses = []
    sys.stdout.flush()


    model.to(device)
    if do_train:
        
        for iter in range(n_iters):
            current_loss=0
            model.train()
            start_time = time.time()
            for j, data in enumerate(train_loader):
                x = data[0].to(device)
                y = data[1].to(device)
                output = model(x)
                loss = criterion(output,y)
                loss.backward()
                optimizer.step()
                current_loss+=loss.item()
                sys.stdout.flush()
            end_time = time.time()
            print(iter, current_loss, end_time-start_time)
            train_losses.append(current_loss)
            if iter % print_every == 0:
                model.eval()
                y_pred = []
                y_true = []
                y_prob= []
                start_time = time.time()
                with torch.no_grad():
                    for v, data in enumerate(val_loader):
                        x = data[0].to(device)
                        y = data[1].to(device)
                        output = model(x)
                        loss = criterion(output,y)
                        probs = torch.sigmoid(output).cpu().detach().numpy().reshape(-1)  # (batch_size, )
                        preds = (probs >=0.5).astype(int)
                        y_pred += list(preds)
                        y_prob +=list(probs)
                        y_true += list(y.cpu().detach().numpy().reshape(-1))
                        current_loss+=loss.item()
                end_time = time.time()
                print('VAL:', iter, current_loss, end_time-start_time)
                val_losses.append(current_loss)
                val_results = utils.eval_dict(y=y_true, 
                                                    y_pred=y_pred, 
                                                    y_prob=y_prob, 
                                                    average='macro',
                                                    thresh_search=True)
                val_results_str = ", ".join(
                    "{}: {:.4f}".format(k, v) for k, v in val_results.items()
                )
                logger.info("VAL - {}".format(val_results_str))
                logger.info("{}".format(classification_report(y_true, y_pred)))
                saver.save(
                        iter, model, optimizer, val_results[metric_name]
                    )
        logger.info("Training DONE.")  


    if do_train:
        import matplotlib
        import matplotlib.pyplot as plt
        matplotlib.use('Agg')
        plt.figure(figsize=(10,8))

        plt.title('Loss')
        plt.plot(train_losses, label='train')
        plt.plot(val_losses, label='val')
        plt.legend(fontsize=30)
        plt.grid()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "graph.png"))

        print('Plotting done')
        sys.stdout.flush()


if __name__ == "__main__":
    encoder_model_development()