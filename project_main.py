# -*- coding: utf-8 -*-

from utils import read_utils
from textgraph import textgraph_read_write

if __name__ == "__main__":
    filenames = read_utils.retrieve_file_names_from_folder('/media/poojaoza/extradrive1/datasets/tg2021-alldata-practice/data/tables/')
    print(filenames)
    # textgraph_read_write.textgraph_data('/media/poojaoza/extradrive1/datasets/tg2021-alldata-practice/data/tables/', 
    #                                     '/media/poojaoza/extradrive1/datasets/tg2021-alldata-practice/data/json_tables/', 
    #                                     filenames)
    textgraph_read_write.process_textgraph_questions('/media/poojaoza/extradrive1/datasets/tg2021-alldata-practice/data/questions.dev.tsv',
                                                     '/media/poojaoza/extradrive1/datasets/tg2021-alldata-practice/data/json_questions/questions.dev.json')
    