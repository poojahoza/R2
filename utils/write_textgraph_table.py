# -*- coding: utf-8 -*-

"""
@author: poojaoza
"""
import json


def convert_textgraph_table_to_format(data):
    
    data_list = []
    for key, val in data.items():
        exp_dict = dict()
        exp_dict['exp_id'] = key
        exp_dict['exp_text'] = val
        data_list.append(exp_dict)
    return data_list

def write_texgraph_table_to_json(data, output_filename):
    
    converted_data = convert_textgraph_table_to_format(data)
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f)
        
def write_textgraph_question_to_json(data, output_filename):
    
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(data, f)