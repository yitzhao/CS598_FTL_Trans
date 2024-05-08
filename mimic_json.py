import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
_DIR = '/scratch/ziyic2/mimic/'
# _DIR = '/mnt/c/code/mimic-iii-clinical-database-1.4/'

df_adm = pd.read_csv(_DIR + 'ADMISSIONS.csv')
df_adm.ADMITTIME = pd.to_datetime(df_adm.ADMITTIME, format = '%Y-%m-%d %H:%M:%S', errors = 'coerce')
df_adm.DISCHTIME = pd.to_datetime(df_adm.DISCHTIME, format = '%Y-%m-%d %H:%M:%S', errors = 'coerce')
df_adm.DEATHTIME = pd.to_datetime(df_adm.DEATHTIME, format = '%Y-%m-%d %H:%M:%S', errors = 'coerce')

df_adm = df_adm.sort_values(['SUBJECT_ID','ADMITTIME'])
df_adm = df_adm.reset_index(drop = True)
df_adm['NEXT_ADMITTIME'] = df_adm.groupby('SUBJECT_ID').ADMITTIME.shift(-1)
df_adm['NEXT_ADMISSION_TYPE'] = df_adm.groupby('SUBJECT_ID').ADMISSION_TYPE.shift(-1)

rows = df_adm.NEXT_ADMISSION_TYPE == 'ELECTIVE'
df_adm.loc[rows,'NEXT_ADMITTIME'] = pd.NaT
df_adm.loc[rows,'NEXT_ADMISSION_TYPE'] = np.NaN

df_adm = df_adm.sort_values(['SUBJECT_ID','ADMITTIME'])

#When we filter out the "ELECTIVE", we need to correct the next admit time for these admissions since there might be 'emergency' next admit after "ELECTIVE"
df_adm[['NEXT_ADMITTIME','NEXT_ADMISSION_TYPE']] = df_adm.groupby(['SUBJECT_ID'])[['NEXT_ADMITTIME','NEXT_ADMISSION_TYPE']].fillna(method = 'bfill')
df_adm['DAYS_NEXT_ADMIT']=  (df_adm.NEXT_ADMITTIME - df_adm.DISCHTIME).dt.total_seconds()/(24*60*60)
df_adm['OUTPUT_LABEL'] = (df_adm.DAYS_NEXT_ADMIT < 30).astype('int')
### filter out newborn and death
df_adm = df_adm[df_adm['ADMISSION_TYPE']!='NEWBORN']
df_adm = df_adm[df_adm.DEATHTIME.isnull()]
df_adm['DURATION'] = (df_adm['DISCHTIME']-df_adm['ADMITTIME']).dt.total_seconds()/(24*60*60)

df_notes = pd.read_csv(_DIR + 'NOTEEVENTS.csv')
df_notes = df_notes.sort_values(by=['SUBJECT_ID','HADM_ID','CHARTDATE'])



df_notes.SUBJECT_ID.unique().shape
df_notes.SUBJECT_ID.shape


charttime_list = df_notes[df_notes.CHARTTIME.notnull()].CHARTTIME.tolist()

charttime_list = sorted(charttime_list)

from random import sample
sample(charttime_list, 10)



df_adm_notes = pd.merge(df_adm[['SUBJECT_ID','HADM_ID','ADMITTIME','DISCHTIME','DAYS_NEXT_ADMIT','NEXT_ADMITTIME','ADMISSION_TYPE','DEATHTIME','OUTPUT_LABEL','DURATION']],
                        df_notes[['SUBJECT_ID','HADM_ID','CHARTDATE','TEXT','CATEGORY', 'CHARTTIME']], 
                        on = ['SUBJECT_ID','HADM_ID'],
                        how = 'left')


# df_adm_notes_list = df_adm_notes.tolist()





df_adm_notes.ADMITTIME_C = df_adm_notes.ADMITTIME.apply(lambda x: str(x).split(' ')[0])
df_adm_notes['ADMITTIME_C'] = pd.to_datetime(df_adm_notes.ADMITTIME_C, format = '%Y-%m-%d', errors = 'coerce')
df_adm_notes['CHARTDATE'] = pd.to_datetime(df_adm_notes.CHARTDATE, format = '%Y-%m-%d', errors = 'coerce')

### If Discharge Summary 
df_discharge = df_adm_notes[df_adm_notes['CATEGORY'] == 'Discharge summary']
# multiple discharge summary for one admission -> after examination -> replicated summary -> replace with the last one 
df_discharge = (df_discharge.groupby(['SUBJECT_ID','HADM_ID']).nth(-1)).reset_index()
df_discharge=df_discharge[df_discharge['TEXT'].notnull()]

df_discharge['CHARTTIME'].isna().sum()


### If Less than n days on admission notes (Early notes)
def less_n_days_data (df_adm_notes, n):
    df_less_n = df_adm_notes[((df_adm_notes['CHARTDATE']-df_adm_notes['ADMITTIME_C']).dt.total_seconds()/(24*60*60))<n]
    df_less_n=df_less_n[df_less_n['TEXT'].notnull()]
    #concatenate first
    df_concat = pd.DataFrame(df_less_n.groupby('HADM_ID')['TEXT'].apply(lambda x: "%s" % ' '.join(x))).reset_index()
    df_concat['OUTPUT_LABEL'] = df_concat['HADM_ID'].apply(lambda x: df_less_n[df_less_n['HADM_ID']==x].OUTPUT_LABEL.values[0])
    return df_concat

df_less_2 = less_n_days_data(df_adm_notes, 2)
df_less_3 = less_n_days_data(df_adm_notes, 3)

import re
def preprocess1(x):
    y=re.sub('\\[(.*?)\\]','',x) #remove de-identified brackets
    y=re.sub('[0-9]+\.','',y) #remove 1.2. since the segmenter segments based on this
    y=re.sub('dr\.','doctor',y)
    y=re.sub('m\.d\.','md',y)
    y=re.sub('admission date:','',y)
    y=re.sub('discharge date:','',y)
    y=re.sub('--|__|==','',y)
    return y

def preprocessing(df_less_n): 
    df_less_n['TEXT']=df_less_n['TEXT'].fillna(' ')
    df_less_n['TEXT']=df_less_n['TEXT'].str.replace('\n',' ')
    df_less_n['TEXT']=df_less_n['TEXT'].str.replace('\r',' ')
    df_less_n['TEXT']=df_less_n['TEXT'].apply(str.strip)
    df_less_n['TEXT']=df_less_n['TEXT'].str.lower()
    df_less_n['TEXT']=df_less_n['TEXT'].apply(lambda x: preprocess1(x))
    #to get 318 words chunks for readmission tasks
    from tqdm import tqdm
    df_len = len(df_less_n)
    want=pd.DataFrame({'ID':[],'TEXT':[],'Label':[]})
    for i in tqdm(range(df_len)):
        x=df_less_n.TEXT.iloc[i].split()
        n=int(len(x)/318)
        for j in range(n):
            want=want._append({'TEXT':' '.join(x[j*318:(j+1)*318]),'Label':df_less_n.OUTPUT_LABEL.iloc[i],'ID':df_less_n.HADM_ID.iloc[i]},ignore_index=True)
        if len(x)%318>10:
            want=want._append({'TEXT':' '.join(x[-(len(x)%318):]),'Label':df_less_n.OUTPUT_LABEL.iloc[i],'ID':df_less_n.HADM_ID.iloc[i]},ignore_index=True)
    return want


def preprocessing_df(df_less_n): 
    df_less_n['TEXT']=df_less_n['TEXT'].fillna(' ')
    df_less_n['TEXT']=df_less_n['TEXT'].str.replace('\n',' ')
    df_less_n['TEXT']=df_less_n['TEXT'].str.replace('\r',' ')
    df_less_n['TEXT']=df_less_n['TEXT'].apply(str.strip)
    df_less_n['TEXT']=df_less_n['TEXT'].str.lower()
    df_less_n['TEXT']=df_less_n['TEXT'].apply(lambda x: preprocess1(x))
    #to get 318 words chunks for readmission tasks
    from tqdm import tqdm
    df_len = len(df_less_n)
    want=pd.DataFrame({'ID':[],'TEXT':[],'Label':[]})
    for i in tqdm(range(df_len)):
        x=df_less_n.TEXT.iloc[i].split()
        n=int(len(x)/318)
        for j in range(n):
            want=want.append({'TEXT':' '.join(x[j*318:(j+1)*318]),'Label':df_less_n.OUTPUT_LABEL.iloc[i],'ID':df_less_n.HADM_ID.iloc[i]},ignore_index=True)
        if len(x)%318>10:
            want=want.append({'TEXT':' '.join(x[-(len(x)%318):]),'Label':df_less_n.OUTPUT_LABEL.iloc[i],'ID':df_less_n.HADM_ID.iloc[i]},ignore_index=True)
    return want


def preprocessing(df_less_n): 
    df_less_n['TEXT']=df_less_n['TEXT'].fillna(' ')
    df_less_n['TEXT']=df_less_n['TEXT'].str.replace('\n',' ')
    df_less_n['TEXT']=df_less_n['TEXT'].str.replace('\r',' ')
    df_less_n['TEXT']=df_less_n['TEXT'].apply(str.strip)
    df_less_n['TEXT']=df_less_n['TEXT'].str.lower()
    df_less_n['TEXT']=df_less_n['TEXT'].apply(lambda x: preprocess1(x))
    #to get 318 words chunks for readmission tasks
    from tqdm import tqdm
    df_len = len(df_less_n)
    want=pd.DataFrame({'ID':[],'TEXT':[],'Label':[]})
    for i in tqdm(range(df_len)):
        x=df_less_n.TEXT.iloc[i].split()
        n=int(len(x)/318)
        for j in range(n):
            want=want._append({
                'ID':df_less_n.HADM_ID.iloc[i],
                'SUBJECT_ID':df_less_n.SUBJECT_ID.iloc[i],
                'TEXT':' '.join(x[j*318:(j+1)*318]),
            'CHARTDATE':df_less_n.CHARTDATE.iloc[i],
            'ADMITTIME':df_less_n.ADMITTIME.iloc[i],
            'Label':df_less_n.OUTPUT_LABEL.iloc[i],
            },
            ignore_index=True)
        if len(x)%318>10:
            want=want._append({
                'ID':df_less_n.HADM_ID.iloc[i],
                'SUBJECT_ID':df_less_n.SUBJECT_ID.iloc[i],
                'TEXT':' '.join(x[j*318:(j+1)*318]),
            'CHARTDATE':df_less_n.CHARTDATE.iloc[i],
            'ADMITTIME':df_less_n.ADMITTIME.iloc[i],
            'Label':df_less_n.OUTPUT_LABEL.iloc[i],
            },ignore_index=True)
    return want



def preprocessing_df(df_less_n): 
    df_less_n['TEXT']=df_less_n['TEXT'].fillna(' ')
    df_less_n['TEXT']=df_less_n['TEXT'].str.replace('\n',' ')
    df_less_n['TEXT']=df_less_n['TEXT'].str.replace('\r',' ')
    df_less_n['TEXT']=df_less_n['TEXT'].apply(str.strip)
    df_less_n['TEXT']=df_less_n['TEXT'].str.lower()
    df_less_n['TEXT']=df_less_n['TEXT'].apply(lambda x: preprocess1(x))
    #to get 318 words chunks for readmission tasks
    from tqdm import tqdm
    df_len = len(df_less_n)
    want=pd.DataFrame({'ID':[],'TEXT':[],'Label':[]})
    for i in tqdm(range(df_len)):
        x=df_less_n.TEXT.iloc[i].split()
        n=int(len(x)/318)
        for j in range(n):
            want=want.append({
                'ID':df_less_n.HADM_ID.iloc[i],
                'SUBJECT_ID':df_less_n.SUBJECT_ID.iloc[i],
                'TEXT':' '.join(x[j*318:(j+1)*318]),
            'CHARTDATE':df_less_n.CHARTDATE.iloc[i],
            'ADMITTIME':df_less_n.ADMITTIME.iloc[i],
            'Label':df_less_n.OUTPUT_LABEL.iloc[i],
            },ignore_index=True)
        if len(x)%318>10:
            want=want.append({
                'ID':df_less_n.HADM_ID.iloc[i],
                'SUBJECT_ID':df_less_n.SUBJECT_ID.iloc[i],
                'TEXT':' '.join(x[j*318:(j+1)*318]),
            'CHARTDATE':df_less_n.CHARTDATE.iloc[i],
            'ADMITTIME':df_less_n.ADMITTIME.iloc[i],
            'Label':df_less_n.OUTPUT_LABEL.iloc[i],
            },ignore_index=True)
    return want

df_discharge = preprocessing_df(df_discharge)

### An example to get the train/test/split with random state:
### note that we divide on patient admission level and share among experiments, instead of notes level. 
### This way, since our methods run on the same set of admissions, we can see the
### progression of readmission scores. 

readmit_ID = df_adm[df_adm.OUTPUT_LABEL == 1].HADM_ID
not_readmit_ID = df_adm[df_adm.OUTPUT_LABEL == 0].HADM_ID
#subsampling to get the balanced pos/neg numbers of patients for each dataset
not_readmit_ID_use = not_readmit_ID.sample(n=len(readmit_ID), random_state=1)
id_val_test_t=readmit_ID.sample(frac=0.2,random_state=1)
id_val_test_f=not_readmit_ID_use.sample(frac=0.2,random_state=1)

id_train_t = readmit_ID.drop(id_val_test_t.index)
id_train_f = not_readmit_ID_use.drop(id_val_test_f.index)

id_val_t=id_val_test_t.sample(frac=0.5,random_state=1)
id_test_t=id_val_test_t.drop(id_val_t.index)

id_val_f=id_val_test_f.sample(frac=0.5,random_state=1)
id_test_f=id_val_test_f.drop(id_val_f.index)

# test if there is overlap between train and test, should return "array([], dtype=int64)"
(pd.Index(id_test_t).intersection(pd.Index(id_train_t))).values

id_test = pd.concat([id_test_t, id_test_f])
test_id_label = pd.DataFrame(data = list(zip(id_test, [1]*len(id_test_t)+[0]*len(id_test_f))), columns = ['id','label'])

id_val = pd.concat([id_val_t, id_val_f])
val_id_label = pd.DataFrame(data = list(zip(id_val, [1]*len(id_val_t)+[0]*len(id_val_f))), columns = ['id','label'])

id_train = pd.concat([id_train_t, id_train_f])
train_id_label = pd.DataFrame(data = list(zip(id_train, [1]*len(id_train_t)+[0]*len(id_train_f))), columns = ['id','label'])

#get discharge train/val/test




# subsampling for training....since we obtain training on patient admission level so now we have same number of pos/neg readmission
# but each admission is associated with different length of notes and we train on each chunks of notes, not on the admission, we need
# to balance the pos/neg chunks on training set. (val and test set are fine) Usually, positive admissions have longer notes, so we need 
# find some negative chunks of notes from not_readmit_ID that we haven't used yet

df = pd.concat([not_readmit_ID_use, not_readmit_ID])
df = df.drop_duplicates(keep=False)
#check to see if there are overlaps
(pd.Index(df).intersection(pd.Index(not_readmit_ID_use))).values

# for this set of split with random_state=1, we find we need 400 more negative training samples
not_readmit_ID_more = df.sample(n=400, random_state=1)
discharge_train_snippets = pd.concat([df_discharge[df_discharge.ID.isin(not_readmit_ID_more)], discharge_train])

#shuffle
discharge_train_snippets = discharge_train_snippets.sample(frac=1, random_state=1).reset_index(drop=True)

#check if balanced
discharge_train_snippets.Label.value_counts()




all_ids = df_discharge.ID.unique()
gather_by_id = {}
for instance in tqdm(df_discharge.iterrows()):
    # print(instance)
    # break
    instance = instance[1]
    if instance['ID'] not in gather_by_id:
        gather_by_id[instance['ID']] = []
    gather_by_id[instance['ID']].append(instance)





import torch
from transformers import AutoTokenizer, BertForSequenceClassification

num_labels = 2
_MODEL = 'emilyalsentzer/Bio_ClinicalBERT'
tokenizer = AutoTokenizer.from_pretrained(_MODEL)



def break_text_into_chunks(text, max_number_of_token=128):
    tokens = tokenizer(text)['input_ids']
    n = len(tokens)
    if n < max_number_of_token:
        return [tokens]
    else:
        chunk_size = max_number_of_token - 2
        chunks = []
        for i in range(0, n, chunk_size):
            chunks.append([101] + tokens[i:i+chunk_size] + [102])
        return chunks


res = {}
for key, list_of_instances in tqdm(gather_by_id.items()):
    # earlist_charttime
    cur = []
    for instance in list_of_instances:
        cur.append(break_text_into_chunks(instance['TEXT']))
    res[key] = {
        'tokens': cur,
        'label': list_of_instances[0]['Label']
    }

# for key in train_id_label.id:
#     assert key in gather_by_id


discharge_train = []
for key in train_id_label.id:
    try:
        discharge_train.append(res[key])
    except:
        pass




discharge_val = []
for key in val_id_label.id:
    try:
        discharge_val.append(res[key])
    except:
        pass


discharge_test = []
for key in test_id_label.id:
    try:
        discharge_test.append(res[key])
    except:
        pass




import json
with open(_DIR + 'discharge/train.json', 'w') as f:
    json.dump(discharge_train, f)

with open(_DIR + 'discharge/val.json', 'w') as f:
    json.dump(discharge_val, f)

with open(_DIR + 'discharge/test.json', 'w') as f:
    json.dump(discharge_test, f)




