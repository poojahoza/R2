#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 04 11:30 2021

@author: poojaoza
"""


train_relevant_json = '/home/poojaoza/Documents/TextGraph2021_preprocessed_data/data/wt-expert-ratings.dev.json'
entity_pairs_txt_file = '/home/poojaoza/Documents/TextGraph2021_preprocessed_data/data/json_tables_entity_pairs_data/json_tables_entity_pairs_train_data.txt'
output_train_entity_pairs_file = '/home/poojaoza/Documents/TextGraph2021_preprocessed_data/data/rbertq1-input-new-candidate-passage-sampling.dev.tsv'
output_train_json_file = '/home/poojaoza/Documents/TextGraph2021_preprocessed_data/data/rbertq1-input-new-candidate-passage-sampling.dev.json'
tables_dir = '/media/poojaoza/extradrive1/projects/tg2021task/data-evalperiod/tables'
candidate_passages_txt_file = '/media/poojaoza/extradrive1/projects/tg2021task/predict-tfidf-dev/predict.txt'



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


train_data_seq_id = 0
for queryid, uuids in candidate_passages_dict.items():
    for item in train_data:
        if item['qid'] == queryid:
            query_text =  "[CLS] "+item['queryText']
            for uid in uuids:
                relevancy = "0"
                for doc in item['documents']:
                    if doc['uuid'] == uid:
                        if int(doc['relevance']) > 0:
                            relevancy = "1"
                if uid in entity_pairs_data:
                    for entity_pair in entity_pairs_data[uid]:
                        qid_uuid = queryid+'*'+uid
                        train_data_query.append(str(train_data_seq_id)+"\t"+query_text+"\t"+entity_pair+"\t"+relevancy+"\t"+qid_uuid+"\n")
                        train_data_seq_id += 1
            break


with open(output_train_entity_pairs_file, 'w') as output_file:
  output_file.writelines(train_data_query)

print(len(train_data_query))

with open(output_train_entity_pairs_file) as out_file:
  out_data = out_file.readlines()

print(len(out_data))
