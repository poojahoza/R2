# -*- coding: utf-8 -*-

from os import listdir
from os.path import isfile, join

def retrieve_file_names_from_folder(mypath):
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    return onlyfiles

def write_textgraph_run_file(ranking_dict, output_file_path):
    #ranking_dict is in the format: {'key1':[1, 2,3], 'key2': [4,5,6]}
    
    string_list = [f'{k}\t{expid}' for k,v in ranking_dict.items() for expid in v]
    
    with open(output_file_path, 'w') as f:
        [f.write(f'{st}\n') for st in string_list]

