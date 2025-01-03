
from utils import CPT_featurization_biannual
from utils import ICD10_featurization_biannual
from utils import med_featurization_biannual
from utils import lab_featurization_biannual
from utils import vitals_featurization_daily

from datetime import datetime, timedelta, date

from source import merge_processed_data

import argparse
import pandas as pd

def base_file_per_day(dfkey, out_file):
    dfkey['admitDate'] = pd.to_datetime(dfkey.ADMIT_DTM).dt.date
    dfkey['dischargeDate'] = pd.to_datetime(dfkey.DISCHARGE_DTM).dt.date
    def to_days(admit, discharge):
        if type(discharge) != pd._libs.tslibs.nattype.NaTType:
            d=(discharge-admit).days
            if d>=0:
                return d
    dfkey['Days in hospital'] = dfkey.apply(lambda x: to_days( x.admitDate, x.dischargeDate), axis=1)

    dfkey = dfkey.loc[(dfkey['Days in hospital'].isna()==False) ]
    print(len(dfkey))



    cols = list(dfkey.columns[9:])+['DayNbr', 'Date']
    dfkey_daily = pd.DataFrame(columns = cols)


    idx = 0
    k= 0
    for i,j in dfkey.iterrows():
        daynbr = 0
        date = j['admitDate']
        while date<=j['dischargeDate']:
            for c in cols:
                if c=='DayNbr':
                    dfkey_daily.at[idx, 'DayNbr'] = daynbr
                elif c=='Date':
                    dfkey_daily.at[idx, 'Date'] = date
                else:
                    dfkey_daily.at[idx, c] = j[c]
            idx+=1
            daynbr +=1
            date = date+timedelta(days=1)
        print(k, daynbr)
        k+=1
        
    dfkey_daily.to_csv(out_file)


def preprocess():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cohort_file', type=str, required=True, help='name of csv file containig all hopitalization events, with patient ID and hospital admission and discharge timestamps ')
    parser.add_argument('--med_file', type=str, required=True, help='name of csv file containig all medications')
    parser.add_argument('--cpt_file', type=str, required=True, help='name of csv file containig all procedure')
    parser.add_argument('--icd_file', type=str, required=True, help='name of csv file containig all conditions')
    parser.add_argument('--lab_file', type=str, required=True, help='name of csv file containig all labs')
    parser.add_argument('--vit_file', type=str, required=True, help='name of csv file containig all vitals')
    
    
    args = parser.parse_args()

    header_data_raw_ehr = 'data/raw/'
    header_data_proc = 'data/'

    print("creating base file with each day represented separately")
    dfkey = pd.read_csv(args.cohort_file)
    base_file_per_day(dfkey, header_data_raw_ehr+"base_file_daily.csv"):

    print('processing medications')
    med_featurization_biannual.medications_feature_vector_biannual(header_data_proc+args.cohort_file, header_data_raw_ehr+args.demo_file, header_data_proc+'med_file.csv')
    print('processing procedures')
    CPT_featurization_biannual.procedures_feature_vector_biannual(header_data_proc+args.cohort_file, header_data_raw_ehr+args.cpt_file, header_data_proc+'cpt_file.csv')
    print('processing conditions')
    ICD10_featurization_biannual.conditions_feature_vector_biannual(header_data_proc+args.cohort_file, header_data_raw_ehr+args.icd_file, header_data_proc+'icd_file.csv')
    print('processing labs')
    lab_featurization_biannual.labs_feature_vector_biannual(header_data_proc+args.cohort_file, header_data_raw_ehr+args.lab_file, header_data_proc+'lab_file.csv')
    print('processing vitals')
    vitals_featurization_daily.vitals_feature_vector_daily(header_data_raw_ehr+"base_file_daily.csv", header_data_raw_ehr+args.lab_file, header_data_proc+'vit_file.csv')



    
    # merge_processed_data.merge_data(header_data_proc+args.cohort_file, 
    #                                 header_data_proc+'demo_file.csv',
    #                                 header_data_proc+'cpt_file.csv',
    #                                 header_data_proc+'icd_file.csv',
    #                                 header_data_proc+'xray_file.csv',
    #                                 header_data_proc+'cohort_file_w_ehr_xray.csv')

if __name__ == "__main__":
    preprocess()   