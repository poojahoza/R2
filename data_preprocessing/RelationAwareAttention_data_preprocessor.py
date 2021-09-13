# -*- coding: utf-8 -*-

import csv


def RelationAwareAttention_data_preprocessor(input_file: str, csv_output_file: str):

    input_train_data = input_file
    final_data_list = []
    sentence_count = 0
    exception_count = 0
    
    
    with open(input_train_data) as reader_file:
      for sentence in reader_file:
        try:
            input_data = []
            
            splited_text = sentence.split('\t')
            query_text = splited_text[1].replace('[CLS]', '').strip()
            explanation_text = splited_text[2].replace('[CLS]', '').strip()
              
            input_data.append(int(splited_text[0]))
            input_data.append(query_text)
            input_data.append(explanation_text)
            input_data.append(int(splited_text[3]))
            input_data.append(splited_text[4])        
        
            final_data_list.append(input_data)
            sentence_count += 1
            print("sentence count : %d " % sentence_count)

        except Exception:
            exception_count += 1
            print("exception count : %d " % exception_count)
    
    print("starting to save to csv")
    writer = csv.writer(open(csv_output_file, 'w'))
    writer.writerow(["sequenceid","query","explanation","relevance","qids_uuids"])
    for row in final_data_list:
        writer.writerow(row)
        print(row[0])

