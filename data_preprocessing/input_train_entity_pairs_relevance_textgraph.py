#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 16:43:16 2021

@author: poojaoza
"""

train_relevant_json = '/home/poojaoza/Documents/TextGraph2021_preprocessed_data/data/wt-expert-ratings.train.json'
entity_pairs_txt_file = '/home/poojaoza/Documents/TextGraph2021_preprocessed_data/data/json_tables_entity_pairs_data/json_tables_entity_pairs_train_data.txt'
output_train_entity_pairs_file = '/home/poojaoza/Documents/TextGraph2021_preprocessed_data/data/rbertq1-input.train.tsv'

# train_relevant_json = '/home/poojaoza/Documents/TextGraph2021_preprocessed_data/data/wt-expert-ratings.dev.json'
# entity_pairs_txt_file = '/home/poojaoza/Documents/TextGraph2021_preprocessed_data/data/json_tables_entity_pairs_data/json_tables_entity_pairs_train_data.txt'
# output_train_entity_pairs_file = '/home/poojaoza/Documents/TextGraph2021_preprocessed_data/data/rbertq1-input.dev.tsv'


import json

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

train_data_seq_id = 0
for item in train_data:
  query_text = "[CLS] "+item['queryText']
  for doc in item['documents']:
    relevancy = "0"
    if doc['isGoldWT21'] == "1":
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