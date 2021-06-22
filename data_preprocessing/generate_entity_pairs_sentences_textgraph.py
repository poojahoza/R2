#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 16:37:36 2021

@author: poojaoza
"""

from os import listdir
from os.path import isfile, join
import json


def generate_entity_pairs_sentences(entity_corpus_path, entity_output_corpus_path):

    entfiles = [f for f in listdir(entity_corpus_path) if isfile(join(entity_corpus_path, f))]
    for f1 in entfiles:
      with open(entity_corpus_path+f1, 'r') as ent_reader_file:
        ent_json_decode = json.load(ent_reader_file)
        final_json = []
        #embed_json_decode = json.load(embed_reader_file)
        for item in ent_json_decode:
          #print(item['WATannotations'][0])
          entity_pairs = []
          if len(item['WATannotations']) == 1:
            exp_paired_text = '[CLS] '+item['exp_text'][:item['WATannotations'][0]["start"]]+"$ "+item['exp_text'][item['WATannotations'][0]["start"]:item['WATannotations'][0]["end"]]+" $ "\
                +item['exp_text'][item['WATannotations'][0]["end"]:]+' == '+item['exp_id']+'\n'
            entity_pairs.append(exp_paired_text)
          else:
            for ann_ind in range(0, len(item['WATannotations'])-1):
              for inner_ann_ind in range(ann_ind+1, len(item['WATannotations'])):
                exp_paired_text = '[CLS] '+item['exp_text'][:item['WATannotations'][ann_ind]["start"]]+"$ "+item['exp_text'][item['WATannotations'][ann_ind]["start"]:item['WATannotations'][ann_ind]["end"]]+" $ "\
                +item['exp_text'][item['WATannotations'][ann_ind]["end"]+1:item['WATannotations'][inner_ann_ind]["start"]]+"# "+item['exp_text'][item['WATannotations'][inner_ann_ind]["start"]:item['WATannotations'][inner_ann_ind]["end"]]+" #"\
                +item['exp_text'][item['WATannotations'][inner_ann_ind]["end"]:]+' == '+item['exp_id']+'\n'
                entity_pairs.append(exp_paired_text)
            #print(entity_pairs)
          item['entity_pairs'] = entity_pairs
          final_json.append(item)
        print(len(final_json))
        with open(entity_output_corpus_path+f1, 'w') as writer_file:
          json.dump(final_json, writer_file)