#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 16:43:16 2021

@author: poojaoza
"""

train_relevant_json = '/home/poojaoza/Documents/TextGraph2021_preprocessed_data/data/wt-expert-ratings.train.json'
entity_pairs_txt_file = '/home/poojaoza/Documents/TextGraph2021_preprocessed_data/data/json_tables_entity_pairs_data/json_tables_entity_pairs_train_data.txt'
output_train_entity_pairs_file = '/home/poojaoza/Documents/TextGraph2021_preprocessed_data/data/rbertq1-input-new-random-sampling.train.tsv'
output_train_json_file = '/home/poojaoza/Documents/TextGraph2021_preprocessed_data/data/rbertq1-input-new-random-sampling.train.json'
tables_dir = '/media/poojaoza/extradrive1/projects/tg2021task/data-evalperiod/tables'

# train_relevant_json = '/home/poojaoza/Documents/TextGraph2021_preprocessed_data/data/wt-expert-ratings.dev.json'
# entity_pairs_txt_file = '/home/poojaoza/Documents/TextGraph2021_preprocessed_data/data/json_tables_entity_pairs_data/json_tables_entity_pairs_train_data.txt'
# output_train_entity_pairs_file = '/home/poojaoza/Documents/TextGraph2021_preprocessed_data/data/rbertq1-input.dev.tsv'


import os
import json
import warnings
from typing import List, Tuple

import pandas as pd

explanations = []

def read_explanations(path: str) -> List[Tuple[str, str]]:
    header = []
    uid = None

    df = pd.read_csv(path, sep='\t', dtype=str)

    for name in df.columns:
        if name.startswith('[SKIP]'):
            if 'UID' in name and not uid:
                uid = name
        else:
            header.append(name)

    if not uid or len(df) == 0:
        warnings.warn('Possibly misformatted file: ' + path)
        return []

    return df.apply(lambda r: (r[uid], ' '.join(str(s) for s in list(r[header]) if not pd.isna(s))), 1).tolist()


for path, _, files in os.walk(tables_dir):
    for file in files:
        explanations += read_explanations(os.path.join(path, file))

if not explanations:
    warnings.warn('Empty explanations')
    
df_e = pd.DataFrame(explanations, columns=('uid', 'text'))

print(df_e['uid'])

train_data = []
entity_pairs_data = dict()

with open(entity_pairs_txt_file) as ept:
  for line in ept:
    split_lines = line.split("==")
    uuid = split_lines[1].strip()
    if uuid in entity_pairs_data:
      entity_pairs_data[uuid].append(split_lines[0].strip())
    else:
      entity_pairs_data[uuid] = [split_lines[0].strip()]

print(len(entity_pairs_data))

with open(train_relevant_json) as train_file:
  train_data = json.load(train_file)

train_data = train_data['rankingProblems']

train_data_query = []

# pos_samples_list_per_query = []
# neg_samples_list_per_query = []


for item in train_data:
    query_text = item['queryText']
    pos_samples = 0
    neg_samples = 0
    r_sample_set = set()
    random_samples = pd.DataFrame()
    doc_set = set()
    for doc in item['documents']:
        doc_set.add(doc['uuid'])
        if int(doc['relevance']) > 0:
            pos_samples += 1
        else:
            neg_samples += 1
    required_neg_samples = neg_samples - pos_samples
    # pos_samples_list_per_query.append(pos_samples)
    # neg_samples_list_per_query.append(neg_samples)
    if required_neg_samples > 0:
        random_samples = df_e.sample(n=required_neg_samples)
    r_sample_set = set(df_e['uid'].tolist())
    common_uuids = doc_set.intersection(r_sample_set)
    if len(common_uuids) > 0:
        for index, r_sample in random_samples.iterrows():
            #print(r_sample)
            r_sample_dict = dict()
            r_sample_dict['relevance'] = 0
            r_sample_dict['uuid'] = r_sample['uid']
            r_sample_dict['docText'] = r_sample['text']
            r_sample_dict['isGoldWT21'] = "0"
            r_sample_dict['goldRole'] = ""
            item['documents'].append(r_sample_dict)
            
# samples_df = pd.DataFrame({'pos':pos_samples_list_per_query, 'neg':neg_samples_list_per_query})
# print(samples_df)

train_data_seq_id = 0
for item in train_data:
  query_text = "[CLS] "+item['queryText']
  for doc in item['documents']:
    relevancy = "0"
    if int(doc['relevance']) > 0:
      relevancy = "1"
    else:
      relevancy = "0"
    if doc['uuid'] in entity_pairs_data:
        for entity_pair in entity_pairs_data[doc['uuid']]:
            qid_uuid = item['qid']+'*'+doc['uuid']
            train_data_query.append(str(train_data_seq_id)+"\t"+query_text+"\t"+entity_pair+"\t"+relevancy+"\t"+qid_uuid+"\n")
            train_data_seq_id += 1

with open(output_train_entity_pairs_file, 'w') as output_file:
  output_file.writelines(train_data_query)

print(len(train_data_query))

with open(output_train_entity_pairs_file) as out_file:
  out_data = out_file.readlines()

print(len(out_data))

with open(output_train_json_file, 'w') as output_json_file:
    output_json_file.dump({'rankingProblems':train_data})