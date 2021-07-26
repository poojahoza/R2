#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 16:43:16 2021

@author: poojaoza
"""

candidate_passages_txt_file = '/media/poojaoza/extradrive1/projects/tg2021task/predict-tfidf-train.txt'
train_relevant_json = '/home/poojaoza/Documents/TextGraph2021_preprocessed_data/data/wt-expert-ratings.train.json'


import json

candidate_passages_dict = dict()
train_relevant_dict = dict()

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
            
with open(train_relevant_json) as train_file:
  train_data = json.load(train_file)

train_data = train_data['rankingProblems']

for item in train_data:
    doc_set = set()
    for doc in item['documents']:
        if int(doc['relevance']) > 0:
            doc_set.add(doc['uuid'])
    train_relevant_dict[item['qid']] = doc_set
    
print(len(train_relevant_dict))
    
for query, ids in train_relevant_dict.items():
    if query in candidate_passages_dict:
        common_ids = len(ids & candidate_passages_dict[query])
        print("qid : {0} common ids: {1} out of {2}".format(query, common_ids, len(ids)))

