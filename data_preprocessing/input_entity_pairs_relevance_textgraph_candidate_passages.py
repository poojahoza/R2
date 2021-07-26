#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 16:43:16 2021

@author: poojaoza
"""

train_relevant_json = '/home/poojaoza/Documents/TextGraph2021_preprocessed_data/data/wt-expert-ratings.dev.json'
entity_pairs_txt_file = '/home/poojaoza/Documents/TextGraph2021_preprocessed_data/data/json_tables_entity_pairs_data/json_tables_entity_pairs_train_data.txt'
candidate_passages_txt_file = '/media/poojaoza/extradrive1/projects/tg2021task/predict-tfidf-dev/predict.txt'
output_train_entity_pairs_file = '/home/poojaoza/Documents/TextGraph2021_preprocessed_data/data/rbertq1-input-tfidf-candidate-passages.dev.tsv'

# train_relevant_json = '/home/poojaoza/Documents/TextGraph2021_preprocessed_data/data/wt-expert-ratings.dev.json'
# entity_pairs_txt_file = '/home/poojaoza/Documents/TextGraph2021_preprocessed_data/data/json_tables_entity_pairs_data/json_tables_entity_pairs_train_data.txt'
# output_train_entity_pairs_file = '/home/poojaoza/Documents/TextGraph2021_preprocessed_data/data/rbertq1-input.dev.tsv'


import json

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
            candidate_passages_dict[queryid].append(split_lines[1].strip())
        else:
            candidate_passages_dict[queryid] = [split_lines[1].strip()]

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
  candidate_passages = candidate_passages_dict[item['qid']]
  relevant_passages = set()
  for document in item['documents']:
      if int(document['relevance']) > 0:
          relevant_passages.add(document['uuid'])
  for uuid in candidate_passages:
    relevancy = "0"
    if uuid in relevant_passages:
      relevancy = "1"
    else:
      relevancy = "0"
    if uuid in entity_pairs_data:
        for entity_pair in entity_pairs_data[uuid]:
            qid_uuid = item['qid']+'*'+uuid
            train_data_query.append(str(train_data_seq_id)+"\t"+query_text+"\t"+entity_pair+"\t"+relevancy+"\t"+qid_uuid+"\n")
            train_data_seq_id += 1

with open(output_train_entity_pairs_file, 'w') as output_file:
  output_file.writelines(train_data_query)

print(len(train_data_query))

with open(output_train_entity_pairs_file) as out_file:
  out_data = out_file.readlines()

print(len(out_data))