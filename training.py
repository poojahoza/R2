#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 14:58:50 2021

@author: poojaoza
"""
import argparse
import torch
import os
import torch.optim as optim
import torch.nn as nn
import numpy as np

from models import RBERTQ1
from transformers import BertConfig
from sklearn.utils.class_weight import compute_class_weight

from data_preprocessing.RBERTQ1_data_preprocessor import RBERTQ1_data_preprocessor


class Training(object):
    
    def __init__(self):
        
        #self.dataset = dataset
        # check if a GPU is present in the machine, if yes then utilize it
        self.device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")
        
        self.config = BertConfig()
        self.model = RBERTQ1(config=self.config, device=self.device).to(self.device)
        #self.class_weights = torch.Tensors([4.5])
        #print(model)
        #self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.class_weights)
        #optimizer = optim.Adam(sample_features, lr=2e-5, )

    def save_model(self, output_model_dir):
        torch.save(self.model, output_model_dir)
        print("model saved")
    
    def load_model(self, model_dir):
        self.model = torch.load(model_dir)
        
    def evaluate(self, evalloader, loss_fn):
        print("Start Evaluation....")
        self.model.eval()
        
        total_loss = 0.0
        total_preds = []
        
        for i, data in enumerate(evalloader):
            print(".")
            
            # Progress update every 500 batches.
            if i % 50 == 0 and not i == 0:
              # Report progress.
              print(' Eval Batch {:>5,}  of  {:>5,}.'.format(i, len(evalloader)))
              
            labels = data[8]
            seqid = data[9]
            data = tuple(d.to(self.device) for i, d in enumerate(data) if i<8)
            
            with torch.no_grad():
                outputs = self.model(data[0], 
                                     data[1], 
                                     data[2], 
                                     data[3], 
                                     data[4], 
                                     data[5], 
                                     data[6], 
                                     data[7])
                #push outputs to cpu
                outputs = outputs.to("cpu")
                loss = loss_fn(outputs, labels.type_as(outputs))
                
                total_loss += loss.item()
                
                outputs = outputs.detach().to('cpu')
                outputs = torch.sigmoid(outputs)
                total_preds.append([outputs, seqid])
        avg_loss = total_loss/len(evalloader)
        #total_preds = np.concatenate(total_preds, axis=0)
        
        return avg_loss, total_preds
    
    def train(self, trainloader, loss_fn, optimizer, output_model_dir):
        self.model.train()
        
        running_loss = 0.0
        total_preds = []
        
        for i, data in enumerate(trainloader):
            print(".")
            #print(i)
            #print(len(data[0]))
            
            # Progress update every 500 batches.
            if i % 50 == 0 and not i == 0:
              # Report progress.
              print(' Train Batch {:>5,}  of  {:>5,}.'.format(i, len(trainloader)))
      
            self.model.zero_grad()
            labels = data[8]
            seqid = data[9]
            data = tuple(d.to(self.device) for i, d in enumerate(data) if i<8)
            outputs = self.model(data[0], 
                                 data[1], 
                                 data[2], 
                                 data[3], 
                                 data[4], 
                                 data[5], 
                                 data[6], 
                                 data[7])
            #push outputs to cpu
            outputs = outputs.to("cpu")
            loss = loss_fn(outputs, labels.type_as(outputs))
            #optimizer.zero_grad()
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            #Using detach().to('cpu') as the utilization of gpu memory increase with
            #every iteration. 
            #Assumed reason: the gradients of the predictions i.e. outputs when appending
            #to the total_preds is causing the increase in utilization of gpu memory
            outputs=outputs.detach().to('cpu')
            total_preds.append([outputs, seqid])
            del outputs
            del labels
            del seqid
            
            running_loss += loss.item()
        self.save_model(output_model_dir)
        #print("Finished training")
        avg_loss = running_loss/len(trainloader)
        return avg_loss, total_preds
        
    
    def train_wrapper(self, train_dataset=None, 
                      output_model_dir='./model', 
                      batchsize=4, 
                      epochs=1, 
                      labels_tensr=None,
                      eval_dataset=None):
        
        print("Started training")
        #num_workers=0 because if it is not zero it causes RuntimeError: received 0 items of ancdata
        #at the below line when using the preprocessed data rbertq1_preprocessed_v1
        #Solution found from https://discuss.pytorch.org/t/runtimeerror-received-0-items-of-ancdata/4999
        # https://github.com/pytorch/pytorch/issues/973
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batchsize, shuffle=True, num_workers=0)  
        #model_parameters = [p for n, p in self.model.named_parameters()]
        
        evalloader = torch.utils.data.DataLoader(eval_dataset, batch_size=batchsize, shuffle=False, num_workers=0)  
        
        # Reshaping the labels tensor from 1 dimension to 2 to calculate the class weights
        x_dim = list(labels_tensr.size())[0]
        reshaped_label_tensr = torch.reshape(labels_tensr,(x_dim,))
        class_weights = compute_class_weight('balanced', np.unique(reshaped_label_tensr), reshaped_label_tensr.numpy())
        class_weights = torch.Tensor([class_weights[1]])
        print(class_weights)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=class_weights)
        optimizer = optim.Adam(self.model.parameters(), lr=2e-5, )
        
        total_epochs = epochs
        
        
        self.model.zero_grad()
        
        train_losses = []
        eval_losses = []
        
        # set initial loss to infinite
        best_valid_loss = float('inf')
        
        
        for epoch in range(total_epochs):
            print("Processing epoch : {0}".format(epoch))
            
            #train model
            train_loss, _ = self.train(trainloader, loss_fn, optimizer, output_model_dir)
            
            #validate model
            eval_loss, _ = self.evaluate(evalloader, loss_fn)
            
            print("Training loss : {0}".format(train_loss))
            print("Evaluation loss : {0}".format(eval_loss))
            
            if eval_loss < best_valid_loss:
                best_valid_loss = eval_loss
                torch.save(self.model, './models/best_models.pth')

            train_losses.append(train_loss)
            eval_losses.append(eval_loss)

            # print("Training loss : {0}".format(train_loss))
            # print("Evaluation loss : {0}".format(eval_loss))
        
        print("Finished training and evaluation")
    
    
def load_preprocessed_data(preprocessed_data_path):
    if os.path.exists(preprocessed_data_path):
      print('in if')
      final_data_list = torch.load(preprocessed_data_path)
    else:
      print('the file path does not exist')
    
    indexed_tokens_tensor = torch.tensor([ind_tokens[0] for ind_tokens in final_data_list])
    segment_ids_tensor = torch.tensor([seg_ids[1] for seg_ids in final_data_list])
    att_mask_tensor = torch.tensor([attn[2] for attn in final_data_list])
    ent1_mask_tensor = torch.tensor([ent1_mask[3] for ent1_mask in final_data_list])
    ent2_mask_tensor = torch.tensor([ent2_mask[4] for ent2_mask in final_data_list])
    query_indexed_tokens_tensor = torch.tensor([q_ind_tokens[5] for q_ind_tokens in final_data_list])
    query_segment_ids_tensor = torch.tensor([q_seg_ids[6] for q_seg_ids in final_data_list])
    query_att_mask_tensor = torch.tensor([q_attn[7] for q_attn in final_data_list])
    labels_tensor = torch.tensor([labels[8] for labels in final_data_list])
    seqid_tensor = torch.tensor([seqid[9] for seqid in final_data_list])
    
    final_dataset = torch.utils.data.TensorDataset(
        indexed_tokens_tensor,
        segment_ids_tensor,
        att_mask_tensor,
        ent1_mask_tensor,
        ent2_mask_tensor,
        query_indexed_tokens_tensor,
        query_segment_ids_tensor,
        query_att_mask_tensor,
        labels_tensor,
        seqid_tensor
    )
    print("Finished loading Preprocessed data")
    print(labels_tensor.shape)
    return final_dataset, labels_tensor

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
    training_parser.add_argument("--trainpreprocessedfile", required=True)
    training_parser.add_argument("--evalpreprocessedfile", required=True)
    training_parser.add_argument("--batchsize", type=int)
    training_parser.add_argument("--epochs", type=int)
    training_parser.add_argument("--save")
    args = parser.parse_args()
    parser_arguments = vars(args)
    
    if parser_arguments['subparser_name'] == "preprocessing":
        dataset = RBERTQ1_data_preprocessor(parser_arguments['input'], 
                                            parser_arguments['output'])
    if parser_arguments['subparser_name'] == "training":
        traindata, trainlabels = load_preprocessed_data(parser_arguments['trainpreprocessedfile'])
        evaldata, labels = load_preprocessed_data(parser_arguments['evalpreprocessedfile'])
        loss, preds = Training().train_wrapper(traindata, 
                                parser_arguments['save'], 
                                parser_arguments['batchsize'],
                                parser_arguments['epochs'],
                                trainlabels,
                                evaldata)
        print(loss)
        print(preds[0])
        