#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 16:43:16 2021

@author: poojaoza
"""


train_relevant_json = input().strip()
entity_pairs_txt_file = input().strip()
output_train_entity_pairs_file = input().strip()
output_train_json_file = input().strip()
tables_dir = input().strip()
candidate_passages_txt_file = input().strip()

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
candidate_passages_dict = dict()

#read the candidate passages file. The format should be textgraphs
#run file i.e. queryid<tab>explanationid
with open(candidate_passages_txt_file) as cp:
    for p in cp:
        split_lines = p.split("\t")
        queryid = split_lines[0].strip()
        if queryid in candidate_passages_dict:
            candidate_passages_dict[queryid].add(split_lines[1].strip())
        else:
            candidate_passages_dict[queryid] = set(split_lines[1].strip())
 
print(len(candidate_passages_dict))

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
    r_sample_set = set()
    random_samples = pd.DataFrame()
    pos_samples = set()
    neg_samples = set()
    for doc in item['documents']:
        if int(doc['relevance']) > 0:
            pos_samples.add(doc['uuid'])
        else:
            neg_samples.add(doc['uuid'])
    pos_difference_uuids = candidate_passages_dict[item['qid']] - pos_samples
    total_diff_uuids = list(pos_difference_uuids - neg_samples)
    random_samples = df_e.loc[df_e['uid'].isin(total_diff_uuids)]
    if len(total_diff_uuids) > 0:
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

# with open(output_train_json_file, 'w') as output_json_file:
#     output_json_file.dump({'rankingProblems':train_data})