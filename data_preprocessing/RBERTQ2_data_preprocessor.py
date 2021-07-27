#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 14:31:57 2021

@author: poojaoza
"""

import torch
import os
import csv

from transformers import BertTokenizer


def RBERTQ2_data_preprocessor(input_file, csv_output_file, features_output_file):

    input_train_data = input_file
    max_sentence_len = 512
    final_data_list = []
    sentence_count = 0
    #sentence_list = []
    exception_count = 0
    
    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    with open(input_train_data) as reader_file:
      for sentence in reader_file:
        #if sentence_count == 5:
          #break
        try:
            input_data = []
        
            # Split the sentence into tokens with BERT tokenizer
            splited_text = sentence.split('\t')
            explanation_text = splited_text[2].replace('[CLS]', '[SEP]')
            tokenized_text = tokenizer.tokenize(splited_text[1]+" "+explanation_text+" [SEP]")
            query_tokenized_len = len(tokenizer.tokenize(splited_text[1]))
            exp_tokenized_len = len(tokenizer.tokenize(splited_text[2]))
            print(tokenized_text)
            #sentence_list.append((splited_text[2], splited_text[3]))
            ent1_pos_st = tokenized_text.index('$')
            ent1_pos_end = tokenized_text.index('$', ent1_pos_st+1)
            ent2_pos_st = tokenized_text.index('#')
            ent2_pos_end = tokenized_text.index('#', ent2_pos_st+1)
            print(ent1_pos_st, ent1_pos_end, ent2_pos_st, ent2_pos_end)
        
            if len(tokenized_text) > max_sentence_len:
              tokenized_text = tokenized_text[:max_sentence_len] # If the length of the sentence is more than max length then truncate
        
            # Map the token strings to their vocabulary indeces.
            indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
            # Mark each of the tokens as belonging to sentence "0".
            segments_ids = [0] * query_tokenized_len + [1] * exp_tokenized_len
            
            # Mask the sentence tokens with 1
            att_mask = [1] * len(indexed_tokens)
        
            # padding the rest of the sequence length
            padding_len = max_sentence_len - len(indexed_tokens)
        
            # Add the padded token to the indexed tokens
            indexed_tokens = indexed_tokens + [0]*padding_len
        
            # Mask the padded tokens with 0
            att_mask = att_mask + [0]*padding_len
        
            # Mark the padded tokens as belonging to sentence "0"
            segments_ids = segments_ids + [0]*padding_len
        
            # Initialize entity masks
            ent1_mask = [0]*len(att_mask)
            ent2_mask = [0]*len(att_mask)
        
            # Mark the entity masks with 1 in the entity positions
            for ent1_ind in range(ent1_pos_st+1, ent1_pos_end):
              ent1_mask[ent1_ind] = 1
            # print(ent1_mask)
        
            for ent2_ind in range(ent2_pos_st+1, ent2_pos_end):
              ent2_mask[ent2_ind] = 1
        
            input_data.append(indexed_tokens)
            input_data.append(segments_ids)
            input_data.append(att_mask)
            input_data.append(ent1_mask)
            input_data.append(ent2_mask)
            input_data.append([int(splited_text[3])])
            input_data.append([int(splited_text[0])])
            input_data.append([splited_text[4]])
        
        
            final_data_list.append(input_data)
            sentence_count += 1
            print("sentence count : %d " % sentence_count)

        except Exception:
            exception_count += 1
            print("exception count : %d " % exception_count)
    
    # print(sentence_count)
    # features_file = trained_model_output_file
    # if os.path.exists(features_file):
    #   print('in if')
    #   final_data_list = torch.load(features_file)
    # else:
    #   torch.save(final_data_list, features_file)
      
    torch.save(final_data_list, features_output_file)
    writer = csv.writer(open(csv_output_file, 'w'))
    writer.writerows(final_data_list)
    
    # indexed_tokens_tensor = torch.tensor([ind_tokens[0] for ind_tokens in final_data_list])
    # segment_ids_tensor = torch.tensor([seg_ids[1] for seg_ids in final_data_list])
    # att_mask_tensor = torch.tensor([attn[2] for attn in final_data_list])
    # ent1_mask_tensor = torch.tensor([ent1_mask[3] for ent1_mask in final_data_list])
    # ent2_mask_tensor = torch.tensor([ent2_mask[4] for ent2_mask in final_data_list])
    # labels_tensor = torch.tensor([labels[5] for labels in final_data_list])
    # seqid_tensor = torch.tensor([seqid[6] for seqid in final_data_list])
    
    # # print(ent1_mask_tensor.shape)
    
    # final_dataset = torch.utils.data.TensorDataset(
    #     indexed_tokens_tensor,
    #     segment_ids_tensor,
    #     att_mask_tensor,
    #     ent1_mask_tensor,
    #     ent2_mask_tensor,
    #     labels_tensor,
    #     seqid_tensor
    # )
    # return final_dataset