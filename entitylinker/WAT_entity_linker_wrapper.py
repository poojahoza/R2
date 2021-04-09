# -*- coding: utf-8 -*-

import json
import argparse
from entitylinker import WAT_entity_linker


def writetofile(output_file, final_dict):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_dict, f)


def fetchfilesfromfolder(input_file):
    final_json = []
    counter = 1
    with open(input_file, 'r', encoding='utf-8') as f:
        json_decode = json.load(f)
        for item in json_decode:
            temp_item = dict()
            print(counter)
            wat_annotations = WAT_entity_linker.wat_entity_linking(item['exp_text'])
            wat_json_list = [w.json_dict() for w in wat_annotations]
            temp_item['exp_id'] = item['exp_id']
            temp_item['exp_text'] = item['exp_text']
            temp_item['WATannotations'] = wat_json_list
            final_json.append(temp_item)
            counter = counter + 1
    return final_json


def wat_linker_wrapper(input, output):
    final_json = fetchfilesfromfolder(input)
    print(len(final_json))
    writetofile(output, final_json)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Please provide input file and output file location")
    parser.add_argument("--i",help="Input JSON file location")
    parser.add_argument("--o",help="Output JSON file location")
    args = parser.parse_args()
    final_json = fetchfilesfromfolder(args.i)
    print(len(final_json))
    writetofile(args.o, final_json)

