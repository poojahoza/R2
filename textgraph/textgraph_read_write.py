# -*- coding: utf-8 -*-

"""
@author: poojaoza
"""

from utils import read_textgraph_table, write_textgraph_table

def textgraph_data(input_folder_path, output_folder_path, filenames_list):
    
    for file in filenames_list:
        file_data = read_textgraph_table.read_textgraph_table(input_folder_path+file)
        filename_without_extension = file[:file.index('.')]
        write_textgraph_table.write_texgraph_table_to_json(file_data, 
                                                           output_folder_path+filename_without_extension+'.json')
        
def process_textgraph_questions(input_filename, output_filename):
    question_data = read_textgraph_table.read_textgraph_questions(input_filename)
    write_textgraph_table.write_textgraph_question_to_json(question_data, output_filename)
