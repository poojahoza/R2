#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 14:15:46 2021

@author: poojaoza
"""

from transformers import BertPreTrainedModel, BertModel
import torch
import torch.nn as nn


class FullyConnectedEntityLayer(nn.Module):
  def __init__(self, input_tensor, output_tensor, device, dropout_rate=0.1):
    super(FullyConnectedEntityLayer, self).__init__()
    self.dropout = nn.Dropout(dropout_rate)
    self.linear = nn.Linear(input_tensor, output_tensor)
    self.activation = nn.Tanh()
    self.linear = self.linear.to(device)
    #self.weight = nn.Parameter(self.linear.weight.grad)

  def forward(self, x):
    x = self.activation(x)
    x = self.linear(x)
    x = self.dropout(x)
    return x

class FullyConnectedCLSLayer(nn.Module):
  def __init__(self, input_tensor, output_tensor, device, dropout_rate=0.1):
    super(FullyConnectedCLSLayer, self).__init__()
    self.dropout = nn.Dropout(dropout_rate)
    self.linear = nn.Linear(input_tensor, output_tensor)
    self.activation = nn.Tanh()
    self.linear = self.linear.to(device)
    #self.weight = nn.Parameter(self.linear.weight.grad)

  def forward(self, x):
    x = self.activation(x)
    x = self.linear(x)
    x = self.dropout(x)
    return x

class FullyConnectedConcatenatedLayer(nn.Module):
  def __init__(self, input_tensor, output_tensor, device, dropout_rate=0.1):
    super(FullyConnectedConcatenatedLayer, self).__init__()
    self.dropout = nn.Dropout(dropout_rate)
    self.linear1 = nn.Linear(input_tensor, output_tensor)
    self.linear1 = self.linear1.to(device)
    #self.weight = nn.Parameter(self.linear1.weight.grad)

  def forward(self, x):
    x = self.linear1(x)
    x = self.dropout(x)
    return x

class RBERTQ1(BertPreTrainedModel):
  def __init__(self, config, device):
    super(RBERTQ1, self).__init__(config)
    self.config = config
    self.d = device
    self.bert = BertModel.from_pretrained('bert-base-uncased',
                                  output_hidden_states = True, # Whether the model returns all hidden-states.
                                  )
    self.config.num_labels = 1
    self.cls_fc_obj = FullyConnectedCLSLayer(self.config.hidden_size, self.config.hidden_size, self.d, self.config.hidden_dropout_prob)
    self.ent_fc_obj = FullyConnectedEntityLayer(self.config.hidden_size, self.config.hidden_size, self.d, self.config.hidden_dropout_prob)
    self.concatenated_fc_obj = FullyConnectedConcatenatedLayer(self.config.hidden_size*3, self.config.num_labels, self.d, self.config.hidden_dropout_prob)
    for param in self.bert.parameters():
         param.requires_grad = False
    
  def forward(self, indexed_tokens, attention_mask, segment_ids, ent1_mask, ent2_mask, query_indexed_tokens, q_attn_mask, q_seg_ids):
    
      
    bert_output = self.bert(indexed_tokens, attention_mask=attention_mask, token_type_ids=segment_ids)
    cls_output = bert_output[1]
    sequence_output = bert_output[0]
    #print(ent1_mask)
    #print(ent2_mask)

    q_bert_output = self.bert(query_indexed_tokens, attention_mask=q_attn_mask, token_type_ids=q_seg_ids)
    q_cls_output = q_bert_output[1]
    #q_sequence_output = q_bert_output[0]

    def entity_average(ent_seq_output, ent_mask):
      
      ent_mask_modified = ent_mask.unsqueeze(1)
      ent_mask_tensor_len = ent_mask.sum(dim=1).unsqueeze(1)
      #print(ent_mask.shape)
      #print(ent_mask_tensor_len)
      sum_tensor = (ent_mask_modified.float() @ ent_seq_output).squeeze(1)
      #print("---------------")
      #print(sum_tensor.float())
      #print(ent_mask_tensor_len.float())
      ent_avg_tensor = sum_tensor.float()/ent_mask_tensor_len.float()
      return ent_avg_tensor

    # Get entity average
    ent1_average_tensor = entity_average(sequence_output, ent1_mask)
    ent2_average_tensor = entity_average(sequence_output, ent2_mask)

    # element-wise multiplication of the query cls output with the sentence cls and entities
    cls_output = cls_output * q_cls_output
    ent1_average_tensor = ent1_average_tensor * q_cls_output
    ent2_average_tensor = ent2_average_tensor * q_cls_output

    
    #softmax = nn.Softmax(dim=1)

    cls_fc_output = self.cls_fc_obj(cls_output)
    ent1_fc_output = self.ent_fc_obj(ent1_average_tensor)
    ent2_fc_output = self.ent_fc_obj(ent2_average_tensor)

    concatenated_output = torch.cat([cls_fc_output, ent1_fc_output, ent2_fc_output], dim=1)
    concatenated_fc_output = self.concatenated_fc_obj(concatenated_output)
    #self.softmax_output = self.softmax(self.concatenated_fc_output)
    #print("==============")
    #print(self.softmax_output.shape)
    #return self.softmax_output
    return concatenated_fc_output

class RBERTQ2(BertPreTrainedModel):
    
  def __init__(self, config, device):
    super(RBERTQ2, self).__init__(config)
    self.config = config
    self.d = device
    self.bert = BertModel.from_pretrained('bert-base-uncased',
                                  output_hidden_states = True, # Whether the model returns all hidden-states.
                                  )
    self.config.num_labels = 1
    self.cls_fc_obj = FullyConnectedCLSLayer(self.config.hidden_size, self.config.hidden_size, self.d, self.config.hidden_dropout_prob)
    self.ent_fc_obj = FullyConnectedEntityLayer(self.config.hidden_size, self.config.hidden_size, self.d, self.config.hidden_dropout_prob)
    self.concatenated_fc_obj = FullyConnectedConcatenatedLayer(self.config.hidden_size*3, self.config.num_labels, self.d, self.config.hidden_dropout_prob)
    for param in self.bert.parameters():
         param.requires_grad = False
    
  def forward(self, indexed_tokens, segment_ids, attention_mask, ent1_mask, ent2_mask):
    bert_output = self.bert(indexed_tokens, attention_mask=attention_mask, token_type_ids=segment_ids)
    cls_output = bert_output[1]
    sequence_output = bert_output[0]
    # print(ent1_mask)
    # print(ent2_mask)

    def entity_average(ent_seq_output, ent_mask):
      
      ent_mask_modified = ent_mask.unsqueeze(1)
      ent_mask_tensor_len = ent_mask.sum(dim=1).unsqueeze(1)
      #print(ent_mask.shape)
      #print(ent_mask_tensor_len)
      sum_tensor = (ent_mask_modified.float() @ ent_seq_output).squeeze(1)
      #print("---------------")
      #print(sum_tensor.float())
      #print(ent_mask_tensor_len.float())
      ent_avg_tensor = sum_tensor.float()/ent_mask_tensor_len.float()
      return ent_avg_tensor

    # Get entity average
    ent1_average_tensor = entity_average(sequence_output, ent1_mask)
    ent2_average_tensor = entity_average(sequence_output, ent2_mask)
    
    # element-wise multiplication of the query cls output with the sentence cls and entities
    cls_output = cls_output
    ent1_average_tensor = ent1_average_tensor
    ent2_average_tensor = ent2_average_tensor
    
    
    #softmax = nn.Softmax(dim=1)
    
    cls_fc_output = self.cls_fc_obj(cls_output)
    ent1_fc_output = self.ent_fc_obj(ent1_average_tensor)
    ent2_fc_output = self.ent_fc_obj(ent2_average_tensor)
    
    concatenated_output = torch.cat([cls_fc_output, ent1_fc_output, ent2_fc_output], dim=1)
    concatenated_fc_output = self.concatenated_fc_obj(concatenated_output)
    #self.softmax_output = self.softmax(self.concatenated_fc_output)
    #print("==============")
    #print(self.softmax_output.shape)
    #return self.softmax_output
    return concatenated_fc_output
