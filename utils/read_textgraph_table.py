# -*- coding: utf-8 -*-

"""
Created on Tue Apr 06 15:08:58 2021

@author: poojaoza
"""

import pandas as pd

def read_textgraph_table(filename):
    
    df=pd.read_csv(filename,sep='\t')
    print(df.count())
    print(list(df.columns))
    df_columns_list = list(df.columns)
    
    if '[SKIP] Comment' in df_columns_list:
        comments_index = df_columns_list.index('[SKIP] Comment')
    elif '[SKIP] COMMENT' in df_columns_list:
        comments_index = df_columns_list.index('[SKIP] COMMENT')
    elif '[SKIP] Comments' in df_columns_list:
        comments_index = df_columns_list.index('[SKIP] Comments')
    else:
        comments_index = df_columns_list.index('[SKIP] COMMENTS')
    
    #print(df_columns_list.index('[SKIP] COMMENTS'))
    print(df_columns_list[comments_index:comments_index+3])
    data = dict()
    
    for index, row in df.iterrows():
        row_data = ''
        for columns in df_columns_list[:comments_index]:
            column_data_str = str(row[columns]).replace(';',' ') 
            if column_data_str != 'nan':
                row_data += ' '+column_data_str
        print(row_data)
        print(row['[SKIP] UID'])
        data[row['[SKIP] UID']] = row_data

    return data

def read_textgraph_questions(filename):
    
    df = pd.read_csv(filename, sep='\t')
    data_list = []
    
    for index, row in df.iterrows():
        
        ques_dict = dict()
        first_index = row['question'].index("(")
        answer_key_index = row['question'].index('('+row['AnswerKey']+')')+len('('+row['AnswerKey']+')')
        last_index = len(row['question'])
        try:
            last_index = row['question'].index('(',answer_key_index)
        except ValueError:
            last_index = len(row['question'])
        ques_dict['question_id'] = row['QuestionID']
        ques_dict['answer_key'] = row['AnswerKey']
        ques_dict['original_question'] = row['question']
        
        # processed_question: contains the question with only the correct answer key
        ques_dict['processed_question'] = row['question'][:first_index]+' '+row['question'][answer_key_index:last_index]
        data_list.append(ques_dict)
        
    return data_list
