#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 14:15:46 2021

@author: poojaoza
"""

from transformers import BertPreTrainedModel, BertModel, BertTokenizer
# from torch_geometric.nn import GATConv
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
from torch import Tensor


class ScaledDotProductionAttention(nn.Module):
    """
    Scaled Dot-Product Attention proposed by "Attention Is All You Need"
    
    Args: dim, mask
        dim (int): dimension of attention
        mask (torch.Tensor): tensor containing indices to be masked
        
    Inputs: query, key, value, mask
        - query (batch, q_len, d_model): tensor containing projection vector
                                         for decoder
        - key (batch, k_len, d_model): tensor containing projection vector for
        encoder
        - value (batch, v_len, d_model): tensor containing features of the 
        encoded input sequence
        - mask (-): tensor containing indices to be masked
        
    Returns: context, attn
        - context: tensor containing the context vector from attention 
        mechanism
        - attention: tensor containing the attention (alignment) from the encoder
        outputs
    """
    
    def __init__(self, dim: int):
        super(ScaledDotProductionAttention, self).__init__()
        self.d_k = np.sqrt(dim)
        
    def forward(self,
                query: Tensor,
                key: Tensor,
                value: Tensor,
                mask: Optional[Tensor]=None) -> Tuple[Tensor, Tensor]:
        
        # let's say the shape of query tensor is (2, 9, 768)
        # and the key tensor shape is (2, 10, 768)
        # the shape of score tensor would be (2, 9, 10)
        score = torch.bmm(query, key.transpose(1,2))/self.d_k
        
        if mask is not None:
            score.masked_fill_(mask.reshape(score.size()), -1e9)
        
        attention = nn.Softmax(score, dim=-1)
        context = torch.bmm(attention, value)
        
        return context, attention



class FullyConnectedEntityLayer(nn.Module):
  def __init__(self, 
               input_tensor: int, 
               output_tensor: int, 
               device: str, 
               dropout_rate:float=0.1):
      
    super(FullyConnectedEntityLayer, self).__init__()
    self.dropout = nn.Dropout(dropout_rate)
    self.linear = nn.Linear(input_tensor, output_tensor, bias=False)
    nn.init.xavier_uniform_(self.linear.weight)
    self.activation = nn.Tanh()
    self.linear = self.linear.to(device)
    

  def forward(self, x: Tensor) -> Tensor:
    x = self.activation(x)
    x = self.linear(x)
    x = self.dropout(x)
    return x

class FullyConnectedCLSLayer(nn.Module):
  def __init__(self, 
               input_tensor: int, 
               output_tensor: int, 
               device: str, 
               dropout_rate: float=0.1):
    super(FullyConnectedCLSLayer, self).__init__()
    self.dropout = nn.Dropout(dropout_rate)
    self.linear = nn.Linear(input_tensor, output_tensor, bias=False)
    nn.init.xavier_uniform_(self.linear.weight)
    self.activation = nn.Tanh()
    self.linear = self.linear.to(device)
    #self.weight = nn.Parameter(self.linear.weight.grad)

  def forward(self, x: Tensor) -> Tensor:
    x = self.activation(x)
    x = self.linear(x)
    x = self.dropout(x)
    return x

class FullyConnectedConcatenatedLayer(nn.Module):
  def __init__(self, 
               input_tensor: int, 
               output_tensor: int, 
               device: str, 
               dropout_rate: float=0.1):
    super(FullyConnectedConcatenatedLayer, self).__init__()
    self.dropout = nn.Dropout(dropout_rate)
    self.linear1 = nn.Linear(input_tensor, output_tensor, bias=False)
    nn.init.xavier_uniform_(self.linear1.weight)
    self.linear1 = self.linear1.to(device)
    #self.weight = nn.Parameter(self.linear1.weight.grad)

  def forward(self, x: Tensor) -> Tensor:
    x = self.linear1(x)
    x = self.dropout(x)
    return x

# class GATSimpleLayer(nn.Module):
    
#   def __init__(self, 
#                config: dict, 
#                device: str, 
#                heads: int):
    
#     super(GATSimpleLayer, self).__init__()

#     self.config = config
#     self.d = device
#     self.heads = heads
#     self.dropout = nn.Dropout(dropout_rate=0.1)
#     self.layer1 = GATConv(self.config.hidden_size, self.config.hidden_size, heads=self.heads, dropout_rate=0.1)
#     self.layer2 = GATConv(self.config.hidden_size, self.config.hidden_size, heads=self.heads, dropout_rate=0.1)
#     self.layer1 = self.layer1.to(self.d)
#     self.layer2 = self.layer2.to(self.d)

#   def forward(self, graph_data: dict) -> Tensor:
#     nodes = graph_data.x
#     edges = graph_data.edge_index

#     nodes = self.layer1(nodes, edges)
#     nodes = nodes.relu()
#     nodes = self.dropout(nodes)
#     nodes = self.layer2(nodes, edges)
#     return nodes

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
    
  def __init__(self, config: dict, device: str):
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
    
  def forward(self, 
              indexed_tokens: Tensor, 
              segment_ids: Tensor, 
              attention_mask: Tensor, 
              ent1_mask: Tensor, 
              ent2_mask: Tensor) -> Tensor:
      
    bert_output = self.bert(indexed_tokens, attention_mask=attention_mask, token_type_ids=segment_ids)
    bert_output_last_hidden_state = bert_output.last_hidden_state
    #cls_output = bert_output[1]
    #sequence_output = bert_output[0]
    cls_output = bert_output_last_hidden_state[:, 0, :]
    sequence_output = bert_output_last_hidden_state
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

class RelationAwareAttention(BertPreTrainedModel):
    
    def __init__(self, config: dict, device: str):
        
        super(RelationAwareAttention, self).__init__(config)
        self.config = config
        self.d = device
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased',
                                  output_hidden_states = True, # Whether the model returns all hidden-states.
                                  )
        
        self.config.num_labels = 1
        self.attention_layer = ScaledDotProductionAttention(768)
        for param in self.bert.parameters():
            param.requires_grad = False
            
    def forward(self,
                query: str,
                key: str,
                value: str,
                mask: Optional[Tensor]=None) -> Tuple[Tensor, Tensor]:
        
        query_tokenizer = self.tokenizer(query, return_tensors="pt", padding=True)
        query_tokenizer.to(self.device)
        query_bert_output = self.bert(**query_tokenizer)
        query_last_hid_layer_output = query_bert_output.last_hidden_state
        
        key_tokenizer = self.tokenizer(key, return_tensors="pt", padding=True)
        key_tokenizer.to(self.device)
        key_bert_output = self.bert(**key_tokenizer)
        key_last_hid_layer_output = key_bert_output.last_hidden_state
        
        value_tokenizer = self.tokenizer(value, return_tensors="pt", padding=True)
        value_tokenizer.to(self.device)
        value_bert_output = self.bert(**value_tokenizer)
        value_last_hid_layer_output = value_bert_output.last_hidden_state
        
        context,attention = self.attention_layer(query_last_hid_layer_output,
                                                 key_last_hid_layer_output,
                                                 value_last_hid_layer_output)
        
        return context, attention
            
        
        
        