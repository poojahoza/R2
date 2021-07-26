# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 15:04:34 2021

@author: poojaoza
"""

import argparse
import json


def get_common_uuids(train_uuids, dev_uuids):
    if len(train_uuids.intersection(dev_uuids)) > 0:
        print(len(train_uuids.intersection(dev_uuids)))
    else:
        print("no common elements")

def get_uuids(input_file):
    with open(input_file, 'r') as f:
        file_data = json.load(f)
    content = file_data['rankingProblems']
    query_uuid_pair = dict()
    uuid_set = set()
    for item in content:
        query_uuid_pair[item["qid"]] = set([doc['uuid'] for doc in item['documents']])
        uuid_set.update([doc['uuid'] for doc in item['documents']])
    return query_uuid_pair, uuid_set
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get common uuids between train and dev ground truth for Textgraphs")
    parser.add_argument('--trainjsonfile')
    parser.add_argument('--devjsonfile')
    args = parser.parse_args()
    print("Started processing train file")
    train_pair, train_uuids = get_uuids(args.trainjsonfile)
    print("Finished processing train file")
    print("Started processing dev file")
    dev_pair, dev_uuids = get_uuids(args.devjsonfile)
    print("Finished processing dev file")
    get_common_uuids(train_uuids, dev_uuids)

