#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 14:58:50 2021

@author: poojaoza
"""
import argparse
import torch
import torch.optim as optim
import torch.nn as nn

from models import RBERTQ1
from transformers import BertConfig

from data_preprocessing.RBERTQ1_data_preprocessor import RBERTQ1_data_preprocessor


class Training(object):
    
    def __init__(self, dataset=None):
        
        self.dataset = dataset
        # check if a GPU is present in the machine, if yes then utilize it
        self.device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")
        self.class_weights = torch.Tensor([4.5])
        
        self.config = BertConfig()
        self.model = RBERTQ1(config=self.config).to(self.device)
        #print(model)
        #self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.class_weights)
        #optimizer = optim.Adam(sample_features, lr=2e-5, )

    def train(self):
        
        print("Started training")
        trainloader = torch.utils.data.DataLoader(self.dataset, batch_size=16, shuffle=True, num_workers=2)  
        model_parameters = [p for n, p in self.model.named_parameters()]
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.class_weights)
        optimizer = optim.Adam(model_parameters, lr=2e-5, )
        running_loss = 0.0
        
        self.model.zero_grad()
        
        for epoch in range(1):
            for i, data in enumerate(trainloader):
                self.model.train()
                data = tuple(d.to(self.device) for d in data)
                outputs = self.model(data[0], 
                                     data[1], 
                                     data[2], 
                                     [1],  
                                     data[3], 
                                     data[4], 
                                     data[5], 
                                     data[6], 
                                     data[7])
                loss = loss_fn(outputs, data[8].type_as(outputs))
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
        print("Finished training")
        return running_loss

# =============================================================================
# # check if a GPU is present in the machine, if yes then utilize it
# device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")
# 
# def training(final_dataset):
#     
#     
#     print("Started with training")
# 
#     trainloader = torch.utils.data.DataLoader(final_dataset, batch_size=16, shuffle=True, num_workers=2)  
#     #classes = ('0', '1')
#     class_weights = torch.Tensor([4.5]).to(device)
#     
#     dataiter = iter(trainloader)
#     sample_features = dataiter.next()
#     #labels = torch.Tensor([1]).to(device)
#     sample_features[0].to(device)
#     sample_features[1].to(device)
#     sample_features[2].to(device)
#     sample_features[3].to(device)
#     sample_features[4].to(device)
#     sample_features[5].to(device)
#     sample_features[6].to(device)
#     sample_features[7].to(device)
#     sample_features[8].to(device)
#     
#     config = BertConfig()
#     model = RBERTQ1(config=config, device=device).to(device)
#     #print(model)
#     loss_fn = nn.BCEWithLogitsLoss(pos_weight=class_weights)
#     optimizer = optim.Adam(sample_features, lr=2e-5, )
#     
#     for epoch in range(1):
#     
#       running_loss = 0.0
#       for i, data in enumerate(trainloader):
#         #inputs = data
#         optimizer.zero_grad()
#     
#         outputs = model(sample_features[0], sample_features[1], sample_features[2], [0, 1],  sample_features[3], sample_features[4], sample_features[5], sample_features[6], sample_features[7])
#         loss = loss_fn(outputs, sample_features[8].type_as(outputs))
#         loss.backward()
#         optimizer.step()
#     
#         running_loss += loss.item()
#     
#         if i % 2000 == 1999:
#           print('[%d, %5d] loss : %.3f' % (epoch+1, i+1, running_loss/2000))
#           running_loss = 0.0
#     
#     print('Finished Training')
# =============================================================================
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="subparser_name")
    preprocessing_parser = subparsers.add_parser("preprocessing")
    training_parser = subparsers.add_parser("training")
    preprocessing_parser.add_argument("--input")
    preprocessing_parser.add_argument("--output")
    training_parser.add_argument("--preprocessedfile", required=True)
    training_parser.add_argument("--save")
    args = parser.parse_args()
    parser_arguments = vars(args)
    
    if parser_arguments['subparser_name'] == "preprocessing":
        dataset = RBERTQ1_data_preprocessor(parser_arguments['input'], 
                                            parser_arguments['output'])
    if parser_arguments['subparser_name'] == "training":
        training(parser_arguments['preprocessedfile'])