# Blood transfusion risk prediction
 GNN model for prediction of risk od blood transfusion among hospitalized patients

## requirements

Place cohort file in data/raw
*   cohort_file -- all hopitalizations (should be 3 days or longer), include demographic data of the patients
Place data files in data/raw/ehr
*   cpt_file.csv
*   icd_file.csv
*   med_file.csv
*   lab_file.csv
*   vit_file.csv


## preprocess
```
python3 code/preprocess.py --cohort_file file_name --med_file file_name --cpt_file file_name --icd_file file_name --lab_file file_name --vit_file file_name
```
## develop temporal embedding model

```
python3 code/TemporalEmbedding.py 
```
Default parameters can be changed in the .py file

## build test graph from scratch

```
python3 code/graph_formation.py 
```
Default parameters can be changed in the .py file

## apply GNN
```
python3 code/GCNN_weighted.py 
```

