import os
import sys
import time
import json
import datetime
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import ray
import modin.pandas as mpd
from tqdm import tqdm
import numpy as np
from transformers import (
    DebertaTokenizer, DebertaForMaskedLM)

abspath = str(Path(__file__).resolve().parent.parent)
sys.path.append(abspath)
from src.utils import pickleload, picklesave, featurization, DotDict
from src.dataset import InferenceDataset
from src.tokenizer import EHRTokenizer
from src.vars import tokenizer_config


abspath = str(Path(__file__).resolve().parent.parent.parent)
dpath = os.path.join(abspath, '../data/mimic/')
cpath = os.path.join(abspath, '../data/concepts')
spath = os.path.join(abspath, 'usedata/mimic/')
tpath = os.path.join(abspath, 'usedata/mimic/tokens')
urpath = os.path.join(abspath, 'usedata/representation')
rpath = os.path.join(abspath, 'results/representation')
model_path = os.path.join(abspath, 'language_models/deberta')


# ECG indexing
print('ECG indexing...')
os.makedirs(os.path.join(abspath, "trashbin/ray_spill"), exist_ok=True)
ray.init(
    num_cpus=64,
    _system_config={
        "object_spilling_config": json.dumps(
            {"type": "filesystem", "params": {
                "directory_path": os.path.join(abspath, "trashbin/ray_spill")
            }}
        )
    },
    object_store_memory=int(200*1024*1024*1024)
)
ecgs = mpd.read_csv(os.path.join(dpath, 'machine_measurements.csv'))
ecgidx = mpd.DataFrame(
    ecgs[[e for e in ecgs.columns if 'report' in e]].values.reshape(-1), 
    columns=['report'])
ecgidx = ecgidx.dropna()['report'].value_counts().reset_index().set_index(['report'])
ecgidx['ecgidx'] = np.arange(len(ecgidx))

# Merging EHR with ECG records and extracting outcomes
print('Merging EHR with ECG records and extracting outcomes...')
ecgidx['idx'] = [f'ecg_{i}' for i in range(len(ecgidx))]
allr = mpd.read_csv(os.path.join(spath, 'allrecords.csv'))
test_id = mpd.read_csv(os.path.join(spath, 'test_id.csv'))
allr = allr[allr['person_id'].isin(test_id['person_id'])]

person = mpd.read_csv(os.path.join(dpath, 'person.csv'))
person['subject_id'] = person['trace_id'].apply(lambda x: x.split(':')[1][:-1]).astype(int)
person = person[['person_id', 'subject_id']].set_index(['subject_id'])

ecgs = mpd.concat(
    [
        mpd.DataFrame(
            ecgs[['subject_id', 'study_id', 'ecg_time', e]].dropna().values, 
            columns=['subject_id', 'study_id', 'ecg_time', 'report']
        ) \
     for e in [e for e in ecgs.columns if 'report' in e]]).sort_values(['subject_id', 'ecg_time'])
ecgs['concept_id'] = ecgidx.loc[ecgs['report'].values]['idx'].values
ecgs = ecgs[ecgs['subject_id'].isin(person.index)]
ecgs['person_id'] = person.loc[ecgs['subject_id']]['person_id'].values
ecgs = ecgs[ecgs['person_id'].isin(test_id['person_id'])]
ecgs['domain'] = 'ecg'
ecgs = ecgs[['person_id', 'domain', 'concept_id', 'ecg_time', 'study_id']]
ecgs.columns = list(allr.columns) + ['study_id']

allr = allr[allr['person_id'].isin(ecgs['person_id'])]
allr['study_id'] = None

ecgs.reset_index(drop=True, inplace=True)
allr.reset_index(drop=True, inplace=True)
allr = mpd.concat([allr, ecgs]).sort_values(['person_id', 'record_datetime'])
allr['record_datetime'] = mpd.to_datetime(allr['record_datetime'])
allr.reset_index(drop=True, inplace=True)
print('Total:', len(set(allr['person_id'])))



# Selecting proper concept IDs for outcomes
# Outcomes: Heart failure, Myocardial infarction, Ischemic stroke, Pulmonary embolism
co = mpd.read_csv(os.path.join(cpath, 'CONCEPT.csv'), delimiter='\t')
cr = mpd.read_csv(os.path.join(cpath, 'CONCEPT_RELATIONSHIP.csv'), delimiter='\t')

base_code = {
    'HF': cr[
                (cr['concept_id_1'].isin(co[
                (co['vocabulary_id'].str.contains('ICD10', na=False)) & 
                (co['concept_code'].str.contains('I50'))]['concept_id'].values)) & 
                (cr['relationship_id'] == 'Maps to')
            ]['concept_id_2'].unique().astype(str),
    'IHD': cr[
                (cr['concept_id_1'].isin(co[
                (co['vocabulary_id'].str.contains('ICD10', na=False)) & 
                (co['concept_code'].str.contains('I20|I21|I24'))]['concept_id'].values)) & 
                (cr['relationship_id'] == 'Maps to')
            ]['concept_id_2'].unique().astype(str),
    'CCAD': cr[
                (cr['concept_id_1'].isin(co[
                (co['vocabulary_id'].str.contains('ICD10', na=False)) & 
                (co['concept_code'].str.contains('I25'))]['concept_id'].values)) & 
                (cr['relationship_id'] == 'Maps to')
            ]['concept_id_2'].unique().astype(str),
    'Stroke': cr[
                (cr['concept_id_1'].isin(co[
                (co['vocabulary_id'].str.contains('ICD10', na=False)) & 
                (co['concept_code'].str.contains('I60|I61|I63'))]['concept_id'].values)) & 
                (cr['relationship_id'] == 'Maps to')
            ]['concept_id_2'].unique().astype(str),
}
all_cvds = np.concatenate(list(base_code.values())).astype(str)

cases = allr[allr['concept_id'].isin(all_cvds)]
case_time = cases.loc[
    cases['person_id'].drop_duplicates(keep='first').index
][['person_id', 'record_datetime']]
case_time['record_datetime'] = mpd.to_datetime(case_time['record_datetime'])
case_time['time_min'] = case_time['record_datetime'] - mpd.DateOffset(days=360)
case_time['time_max'] = case_time['record_datetime'] - mpd.DateOffset(days=30)
case_df = mpd.merge(
    allr,
    case_time[['person_id', 'time_min', 'time_max']],
    on='person_id', how='inner'
)
case_df = case_df[
    (case_df['domain'] == 'ecg') & 
    (case_df['record_datetime'].between(case_df['time_min'], case_df['time_max']))
]
case_df = case_df.loc[
    case_df['person_id'].drop_duplicates(keep='last').index
    ][['person_id', 'record_datetime', 'study_id']]
case_df.columns = ['person_id', 'index_time', 'study_id']
print('between 30 and 365 days:', len(set(case_df['person_id'])))


# Control index date
control = allr[~allr['person_id'].isin(cases['person_id'])].reset_index(drop=True)
print('control number:', len(set(control['person_id'])))
last_rec = control.loc[
    control['person_id'].drop_duplicates(keep='last').index
    ][['person_id', 'record_datetime']]
last_rec['last-1year'] = last_rec['record_datetime'] - mpd.DateOffset(days=365)
control = mpd.merge(
    control,
    last_rec[['person_id', 'last-1year']],
    on='person_id', how='inner'
)
control = control[
    (control['record_datetime'] <= control['last-1year']) & 
    (control['domain'] == 'ecg')
]
control_df = control.loc[
    control['person_id'].drop_duplicates(keep='last').index
    ][['person_id', 'record_datetime', 'study_id']]
control_df.columns = ['person_id', 'index_time', 'study_id']
print('between 30 and 365 days:', len(set(control_df['person_id'])))

alldf = []
for n, df in enumerate((case_df, control_df, )):
    use_df = mpd.merge(
        allr[allr.columns.difference(['study_id'], sort=False)],
        df,
        on='person_id', how='inner'
    )
    use_df = use_df[use_df['record_datetime'] <= use_df['index_time']]
    casecnt = use_df[use_df['domain'] != 'ecg']['person_id'].value_counts()
    use_df = use_df[use_df['person_id'].isin(
        casecnt[casecnt >= 30].index
    )]
    use_df = use_df[['person_id', 'index_time', 'study_id']].drop_duplicates()
    alldf.append(use_df)
    print('more than 30 records:', len(set(use_df['person_id'])))

alldf = mpd.concat(alldf)
pidx = mpd.Series(person.index, index=person['person_id'])
alldf['subject_id'] = pidx.loc[alldf['person_id']].values

allr_label = mpd.merge(
    allr[['person_id', 'concept_id', 'record_datetime']], case_df,
    on=['person_id'], how='inner'
)
allr_label = allr_label[allr_label['record_datetime'].between(
    allr_label['index_time'] + mpd.DateOffset(days=30),
    allr_label['index_time'] + mpd.DateOffset(days=360),
    inclusive='both'
)]
for k, v in base_code.items():
    allr_label[k] = allr_label['concept_id'].isin(v).astype(int)
allr_label = allr_label.groupby('person_id')[['HF', 'IHD', 'CCAD', 'Stroke']].any()

alldf = mpd.merge(
    alldf, allr_label.reset_index(),
    on=['person_id'], how='left'
).fillna(False)
for k in base_code.keys():
    alldf[k] = alldf[k].astype(int)
alldf = alldf.sort_values(['person_id'])    
# alldf.to_csv(os.path.join(spath, 'ecg_index_time.csv'), index=None)


# Merge allrecords and visit
print('Assigning additional tokens...')
print('Merge allrecords and visit...')
visit = mpd.read_csv(os.path.join(dpath, 'visit_occurrence.csv'))
allrecords = allr[['person_id', 'concept_id', 'record_datetime', 'domain']]
visit = visit[['person_id', 'visit_concept_id', 'visit_start_datetime']].drop_duplicates()
visit['domain'] = 'visit'
visit.columns = ['person_id', 'concept_id', 'record_datetime', 'domain']
visit = visit[['person_id', 'record_datetime', 'domain', 'concept_id']]
visit = visit[visit['person_id'].isin(allrecords['person_id'])]
allrecords = mpd.concat([allrecords, visit]).reset_index(drop=True)
allrecords['record_datetime'] = mpd.to_datetime(allrecords['record_datetime'])
print('Done')

# Age setting
print('Age setting...')
patients = mpd.read_csv(os.path.join(dpath, 'patients.csv'))
person = mpd.read_csv(os.path.join(dpath, 'person.csv'))
person['subject_id'] = person['trace_id'].apply(lambda x: x.split(':')[1][:-1]).astype(int)
person = mpd.merge(
    person, patients[['subject_id', 'anchor_age']],
    on='subject_id', how='left'
)
person['year_of_birth'] = person['year_of_birth'] - person['anchor_age']

allrecords = mpd.merge(
    allrecords,
    person[['person_id', 'year_of_birth']],
    on=['person_id'], how='left'
)
allrecords['record_year'] = mpd.to_datetime(allrecords['record_datetime']).dt.year
allrecords['age'] = allrecords['record_year'] - allrecords['year_of_birth']
allrecords['concept_id'] = allrecords['concept_id'].astype(str)
allrecords = allrecords[['person_id', 'concept_id', 'record_datetime', 'domain', 'age']]
print('Done')

# Assign visit and domain index
print('Assign visit and domain index...')
allrecords = allrecords.sort_values(['person_id', 'record_datetime'])
ar_visit = allrecords[allrecords['domain'] == 'visit'][['person_id', 'record_datetime']].sort_values(
    ['person_id', 'record_datetime']
)
ar_visit['record_date'] = mpd.to_datetime(ar_visit['record_datetime']).dt.date
ar_visit['row'] = (ar_visit.groupby(['person_id', 'record_date']).cumcount()+1).iloc[:, 0] # modin.pandas bug
ar_visit = ar_visit[ar_visit['row'] == 1].drop(columns=['row'])
ar_visit['visit_rank'] = ar_visit.groupby('person_id')['record_date'].rank(method='dense')
allrecords.loc[ar_visit.index, 'visit_rank'] = list(ar_visit['visit_rank'].astype(int))
allrecords['visit_rank'] = allrecords.groupby(
    'person_id')['visit_rank'].apply(lambda x: x.bfill().ffill()).values

domain_idx = ['special_token', 'cond', 'drug', 'meas', 'proc', 'ecg', 'visit']
allrecords['domain'] = allrecords['domain'].apply(lambda x: domain_idx.index(x))
allrecords['concept_id'] = allrecords['concept_id'].astype(str)
allrecords = allrecords[allrecords['domain'] != 6] # Omit visit records
allrecords['record_rank'] = allrecords.groupby(
    ['person_id', 'visit_rank'])['record_datetime'].rank(method='dense')
allrecords.to_csv(os.path.join(spath, 'allrecords_ecg.csv'), index=None)
print('Done')

print('Save vocab...')
vocab = torch.load(os.path.join(spath, 'vocab.pt'))
curr_cnt = len(vocab)
for n, e in enumerate(ecgidx['idx']):
    vocab[e] = curr_cnt + n
# torch.save(vocab, os.path.join(spath, 'vocab_ecg.pt'))
# with open(os.path.join(spath, 'vocab_ecg.json'), 'w') as f:
#     json.dump(vocab, f)
print('Done')


# Tokenize for finetuning
genders = person[['person_id', 'gender_concept_id']]
genders['gender_source_value'] = person['gender_concept_id'].apply(lambda x: 'M' if x == 8507 else 'F')
genders = genders[['person_id', 'gender_source_value']].set_index('person_id')
alldf['label'] = list(alldf[list(base_code.keys())].values)

allrecords_noecg = allrecords[allrecords['domain'] != 5]
allr_dict = {
    'ecg': allrecords,
    'noecg': allrecords_noecg
}
os.makedirs(os.path.join(tpath, 'Finetuning'), exist_ok=True)
for name, _allr in allr_dict.items():
    savepath = os.path.join(tpath, 'Finetuning', f'CVD_mimic_{name}.pkl')
    print('Save Path: ', savepath)

    o_df = mpd.merge(
        _allr,
        alldf[['person_id', 'index_time', 'label']],
        on='person_id', how='left'
    )
    o_df = o_df[o_df['index_time'].notnull()]
    o_df = o_df[o_df['record_datetime'] <= o_df['index_time']]
    userecords = o_df.sort_values(
        ['person_id', 'record_datetime'], ascending=[True, False])
    if name == 'ecg':
        noecg_records = userecords[userecords['domain'] != 5]
        noecg_records['erow'] = noecg_records.groupby(['person_id']).cumcount()+1
        userecords['row'] = np.nan
        userecords.loc[noecg_records.index, 'row'] = noecg_records['erow'].values
        userecords['row'] = userecords['row'].fillna(method='bfill')
    else:
        userecords['row'] = userecords.groupby(['person_id']).cumcount()+1
    userecords = userecords[userecords['row'] <= 2048].sort_values([
        'person_id', 'record_datetime'
    ])
        
    userecords['visit_rank'] = userecords.groupby(
        'person_id')['visit_rank'].transform(lambda x: x - x.min() + 1)
    userecords['record_rank'] = userecords.groupby(
        ['person_id', 'visit_rank'])['record_datetime'].rank(method='dense')
    tokenizer_config = DotDict({
        'sep_tokens': False, # should we add [SEP] tokens?
        'cls_token': True, # should we add a [CLS] token?
        'padding': True, # should we pad the sequences?
        'truncation': userecords['person_id'].value_counts().max()}) # how long should the longest sequence be
    tokenizer = EHRTokenizer(vocabulary=vocab, config=tokenizer_config)
    
    
    gender_info = genders.values.reshape(-1)
    features, labels = featurization(userecords, gender_info, True)
    print('Done')
    print('Tokenizing...')

    tokenizer.freeze_vocabulary()
    tokenized = tokenizer(features)
    picklesave((tokenized, labels), savepath)
