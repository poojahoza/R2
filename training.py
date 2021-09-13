#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 14:58:50 2021

@author: poojaoza
"""
import argparse
import torch
import os
import random
import json
import torch.optim as optim
import torch.nn as nn
import numpy as np
import pandas as pd
import write_utils

from models import RBERTQ1, RBERTQ2, RelationAwareAttention
from transformers import BertConfig
from sklearn.utils.class_weight import compute_class_weight

from data_preprocessing.RBERTQ1_data_preprocessor import RBERTQ1_data_preprocessor
from data_preprocessing.RBERTQ2_data_preprocessor import RBERTQ2_data_preprocessor
from data_preprocessing.RelationAwareAttention_data_preprocessor import RelationAwareAttention_data_preprocessor
from data_loaders.TextGraphDatasetLoader import TextGraphDatasetLoader
from data_loaders.TextGraphCandidateDatasetLoader import TextGraphCandidateDatasetLoader


   
class Training(object):
    
    def __init__(self, experiment):
        
        #self.dataset = dataset
        # check if a GPU is present in the machine, if yes then utilize it
        self.device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")
        
        self.set_seed()
        
        self.config = BertConfig()
        self.experiment = experiment
        
        if experiment == "RBERTQ2":
            self.model = RBERTQ2(config=self.config, device=self.device)
        else:
            self.model = RelationAwareAttention(config=self.config, device=self.device)
        
        self.model.to(self.device)
        

    def set_seed(self):
        torch.manual_seed(3) #setting RNG for all devices (CPU and CUDA)
        torch.cuda.manual_seed_all(3) #setting RNF across all GPUs
        np.random.seed(3)
        random.seed(3)

    def save_model(self, output_model_dir):
        torch.save(self.model, output_model_dir)
        print("model saved")
    
    def load_model(self, model_dir):
        self.model = torch.load(model_dir)

    def ranking(self, model_output_data, output_predict_file):
        ranking_dict = dict()
        final_ranking_dict = dict()
        for pred in model_output_data:
            output_probs = pred[0].tolist()
            # sequence_ids = pred[1].tolist()
            uids = pred[1]
            for index, ids in enumerate(output_probs):
                # uid = uids_dict[ids[0]]
                query_id, exp_ids = uids[index].split('*')
                output_relevancy = 0
                if output_probs[index] > 0.5:
                    output_relevancy = 1
                if query_id in ranking_dict:
                    temp_dict = ranking_dict[query_id]
                    if exp_ids in temp_dict:
                        ranking_dict[query_id][exp_ids] += output_relevancy
                    else:
                        ranking_dict[query_id][exp_ids] = output_relevancy
                else:
                    ranking_dict[query_id] = {exp_ids: output_relevancy}
        for query, exps in ranking_dict.items():
            sorted_inner_dict = dict(sorted(exps.items(), key=lambda item: item[1], reverse= True)[:100])
            final_ranking_dict[query] = [*sorted_inner_dict.keys()]
        return final_ranking_dict
        
    def test(self, train_dataset, saved_model, batchsize):
        self.load_model(saved_model)
        print(self.model)
        self.model = self.model.to(self.device)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batchsize, shuffle=False, num_workers=0)  
        #model_parameters = [p for n, p in self.model.named_parameters()]
        
        #eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=batchsize, shuffle=False, num_workers=0)  
        self.model.eval()
        
        total_preds = []
        test_batch_no = 1
        
        for eval_data, eval_labels, eval_uids in train_dataloader:
            print(".")
            
            # Progress update every 500 batches.
            if test_batch_no % 50 == 0 and not test_batch_no == 0:
              # Report progress.
              print(' Eval Batch {:>5,}  of  {:>5,}.'.format(test_batch_no, len(train_dataloader)))
              
            # labels = data[8]
            # seqid = data[9]
            eval_data = tuple(d.to(self.device) for d in eval_data)
            
            
            
            with torch.no_grad():
                
                if self.experiment == "RBERTQ2":
                
                    outputs = self.model(eval_data[0], 
                                         eval_data[1], 
                                         eval_data[2], 
                                         eval_data[3], 
                                         eval_data[4])
                elif self.experiment == "RelationAwareAttention":
                    
                    outputs, attnt = self.model(eval_data[0], 
                                         eval_data[1], 
                                         eval_data[1])
                # outputs = outputs.detach().to('cpu')
                # outputs = torch.sigmoid(outputs)
                # total_preds.append([outputs, seqid, labels])
                squeezed_output = torch.squeeze(outputs)
                squeezed_output = squeezed_output.to(self.device)
                eval_labels = eval_labels.to(self.device)
                
                squeezed_output = torch.sigmoid(squeezed_output)
                total_preds.append([squeezed_output, eval_uids, eval_labels])
                test_batch_no += 1
        
        #total_preds = np.concatenate(total_preds, axis=0)
        
        return total_preds

    
    def evaluate(self, evalloader, loss_fn, eval_correct):
        print("Start Evaluation....")
        self.model.eval()
        
        total_loss = 0.0
        total_preds = []
        eval_batch_no = 1
        
        for eval_data, eval_labels, eval_uids in evalloader:
            #print(".")
            
            # Progress update every 500 batches.
            if eval_batch_no % 50 == 0 and not eval_batch_no == 0:
              # Report progress.
              print(' Eval Batch {:>5,}  of  {:>5,}.'.format(eval_batch_no, len(evalloader)))
              
            # labels = data[8]
            # seqid = data[9]
            # data = tuple(d.to(self.device) for i, d in enumerate(data) if i<8)

            
            
            with torch.no_grad():
                
                if self.experiment == "RBERTQ2":
                    eval_data = tuple(d.to(self.device) for d in eval_data)
                    outputs = self.model(eval_data[0], 
                                         eval_data[1], 
                                         eval_data[2], 
                                         eval_data[3], 
                                         eval_data[4])
                elif self.experiment == "RelationAwareAttention":
                    eval_data = tuple(list(d).to(self.device) for d in eval_data)
                    outputs, attnt = self.model(eval_data[0],
                                                eval_data[1],
                                                eval_data[1])
                #push outputs to cpu
                
                # outputs = outputs.to("cpu")

                squeezed_output = torch.squeeze(outputs)
                squeezed_output = squeezed_output.to(self.device)
                eval_labels = eval_labels.to(self.device)

                loss = loss_fn(squeezed_output, eval_labels.type_as(squeezed_output))
                
                total_loss += loss.item()
                
                # squeezed_output = squeezed_output.detach().to('cpu')
                squeezed_output = torch.sigmoid(squeezed_output)
                total_preds.append([squeezed_output, eval_uids, eval_labels])
                predicated_values = (squeezed_output>0.5).float()
                true_values = eval_labels.float()
                eval_correct += (predicated_values == true_values).float().sum()

                eval_batch_no += 1

        avg_loss = total_loss/len(evalloader)
        #total_preds = np.concatenate(total_preds, axis=0)
        
        return avg_loss, total_preds, eval_correct
    
    def train(self, trainloader, loss_fn, optimizer, output_model_dir, batch_train_losses, train_correct, ep_no):
        self.model.train()
        
        running_loss = 0.0
        total_preds = []
        batch_no = 1
        
        for train_data, train_labels, train_uids in trainloader:
            #print(i)
            #print(len(data[0]))
            
            # Progress update every 500 batches.
            # if i % 50 == 0 and not i == 0:
            #   # Report progress.
            #   print(' Train Batch {:>5,}  of  {:>5,}.'.format(i, len(trainloader)))
      
            self.model.zero_grad()
            
            # labels = data[8]
            # seqid = data[9]
            # data = tuple(d.to(self.device) for i, d in enumerate(data) if i<8)
            # outputs = self.model(data[0], 
            #                      data[1], 
            #                      data[2], 
            #                      data[3], 
            #                      data[4], 
            #                      data[5], 
            #                      data[6], 
            #                      data[7])
            
            if self.experiment == 'RBERTQ2':
                
                train_data = tuple(d.to(self.device) for d in train_data)
            
                outputs = self.model(train_data[0],
                                     train_data[1],
                                     train_data[2],
                                     train_data[3],
                                     train_data[4])
            elif self.experiment == "RelationAwareAttention":
                train_data = tuple(list(d).to(self.device) for d in train_data)
                outputs, attnt = self.model(train_data[0],
                                            train_data[1],
                                            train_data[1])
            #push outputs to cpu
            #outputs = outputs.to("cpu")
            #loss = loss_fn(outputs, labels.type_as(outputs))

            outputs = outputs.to(self.device)
            squeezed_output = torch.squeeze(outputs)
            squeezed_output = squeezed_output.to(self.device)
            train_labels = train_labels.to(self.device)
            loss = loss_fn(squeezed_output, train_labels.type_as(squeezed_output))
            print("batch : {0} loss : {1}".format(batch_no, loss))
            #batch_train_losses = batch_train_losses.to('cpu')
            #batch_train_losses.append([i,loss])
            #optimizer.zero_grad()
            loss.backward()

            #print(self.model.state_dict())

            #plot_grad_flow(self.model.named_parameters(), i, ep_no, self.model.parameters())
            #torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()

            
            #Using detach().to('cpu') as the utilization of gpu memory increase with
            #every iteration. 
            #Assumed reason: the gradients of the predictions i.e. outputs when appending
            #to the total_preds is causing the increase in utilization of gpu memory
            #squeezed_output=squeezed_output.detach().to('cpu')

            #train_labels=train_labels.detach().to('cpu')

            # total_preds.append([outputs, seqid, labels])

            

            squeezed_output = torch.sigmoid(squeezed_output)
            # print("maxmimum : {0} minimum : {1} ".format(torch.max(squeezed_output).item(), torch.min(squeezed_output).item()))

            predicated_values = (squeezed_output>0.5).float()
            # true_values = labels.float()
            
            predicated_values_relevant = (predicated_values == 1.0).float().sum()
            # print(predicated_values_relevant.item())

            total_preds.append([predicated_values, train_uids, train_labels])

            true_values = train_labels.float()
            true_values_relevant = (true_values == 1.0).float().sum()
            # print(true_values_relevant.item())
            common_relevant_predicts = 0
            # print(predicated_values)
            for pred_val, true_val in zip(predicated_values, true_values):
              # print(pred_val.item(), true_val.item())
              if pred_val.item() == true_val.item() and pred_val.item() == 1.0:
                common_relevant_predicts += 1
            train_correct += (predicated_values == true_values).float().sum()

            batch_train_losses.append([batch_no, predicated_values_relevant.item(), true_values_relevant.item(), common_relevant_predicts])
            del outputs
            del train_labels
            del squeezed_output
            del train_uids
            
            running_loss += loss.item()
            batch_no += 1
        self.save_model(output_model_dir)
        #print("Finished training")
        #pd.DataFrame(batch_train_losses).to_csv('1peoch_batch_lossses.csv', header=None, index=None)
        avg_loss = running_loss/len(trainloader)
        return avg_loss, total_preds, train_correct
        
    
    def train_wrapper(self, train_dataset=None, 
                      output_model_dir='./model', 
                      batchsize=4, 
                      epochs=1, 
                      eval_dataset=None):
        
        print("Started training")
        #num_workers=0 because if it is not zero it causes RuntimeError: received 0 items of ancdata
        #at the below line when using the preprocessed data rbertq1_preprocessed_v1
        #Solution found from https://discuss.pytorch.org/t/runtimeerror-received-0-items-of-ancdata/4999
        # https://github.com/pytorch/pytorch/issues/973
        
        if self.experiment == "RBERTQ2":
            labels_tensr = torch.tensor([json.loads(label)[0] for label in train_dataset.y])
        elif self.experiment == "RelationAwareAttention":
            labels_tensr = torch.tensor([label for label in train_dataset.y])
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batchsize, shuffle=True, num_workers=0)  
        #model_parameters = [p for n, p in self.model.named_parameters()]
        
        evalloader = torch.utils.data.DataLoader(eval_dataset, batch_size=batchsize, shuffle=False, num_workers=0)  
        
        # Reshaping the labels tensor from 1 dimension to 2 to calculate the class weights
        reshaped_label_tensr = torch.reshape(labels_tensr,(labels_tensr.shape[0],))
        class_weights = compute_class_weight('balanced', np.unique(reshaped_label_tensr), reshaped_label_tensr.numpy())
        # print(class_weights)
        class_weights = torch.Tensor([class_weights[1]])
        # print(class_weights)
        class_weights = class_weights.to(self.device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=class_weights)
        #loss_fn = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=2e-5)
        
        total_epochs = epochs
        
        
        self.model.zero_grad()
        
        train_losses = []
        eval_losses = []
        
        train_correct_list = []
        eval_correct_list = []
        
        # set initial loss to infinite
        best_valid_loss = float('inf')
        
        
        for epoch in range(total_epochs):
            print("Processing epoch : {0}".format(epoch))
            #print(str(self.model))
            self.set_seed()

            train_correct = 0
            eval_correct = 0
            batch_train_losses = []
            
            #train model
            train_loss, train_preds, train_correct = self.train(trainloader, loss_fn, optimizer, output_model_dir, batch_train_losses, train_correct, epoch)

            df = pd.DataFrame(batch_train_losses)
            df.to_csv('prediction_truth_values_batchwise_'+str(epoch)+'.csv', index=False, header=False)

            print("length of train losses {0} ".format(len(batch_train_losses)))
            
            #validate model
            eval_loss, eval_preds, eval_correct = self.evaluate(evalloader, loss_fn, eval_correct)
            
            train_accuracy = 100*train_correct/len(train_dataset)
            eval_accuracy = 100*eval_correct/len(eval_dataset)
            
            print("Training loss : {0} , accuracy : {1}".format(train_loss, train_accuracy))
            print("Evaluation loss : {0}, accuracy : {1}".format(eval_loss, eval_accuracy))
            
            
            #if eval_loss < best_valid_loss:
                #best_valid_loss = eval_loss
                #torch.save(self.model, './models/best_models.pth')

            train_losses.append(train_loss)
            eval_losses.append(eval_loss)
            train_correct_list.append(train_accuracy)
            eval_correct_list.append(eval_accuracy)

            print("Training loss : {0}".format(train_loss))
            print("Evaluation loss : {0}".format(eval_loss))

            print(train_losses)
            print(eval_losses)
            print(train_correct_list)
            print(eval_correct_list)
        
        print(train_losses)
        print(eval_losses)
        print(train_correct_list)
        print(eval_correct_list)
        print("Finished training and evaluation")

        return train_preds

 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="subparser_name")
    preprocessing_parser = subparsers.add_parser("preprocessing")
    training_parser = subparsers.add_parser("training")
    ranking_parser = subparsers.add_parser("ranking")
    preprocessing_parser.add_argument("--preprocessmodel")
    preprocessing_parser.add_argument("--input")
    preprocessing_parser.add_argument("--csvoutput")
    preprocessing_parser.add_argument("--featuresoutput")
    training_parser.add_argument("--trainpreprocessedfile", required=True)
    training_parser.add_argument("--evalpreprocessedfile", required=True)
    training_parser.add_argument("--batchsize", type=int)
    training_parser.add_argument("--epochs", type=int)
    training_parser.add_argument("--save")
    training_parser.add_argument("--experiment", required=True)
    ranking_parser.add_argument("--model")
    ranking_parser.add_argument("--trainpreprocessedfile")
    ranking_parser.add_argument("--evalpreprocessedfile")
    ranking_parser.add_argument("--rankingoutput")
    ranking_parser.add_argument("--batchsize", type=int)
    args = parser.parse_args()
    parser_arguments = vars(args)
    
    if parser_arguments['subparser_name'] == "preprocessing":
        if parser_arguments['preprocessmodel'] == "RBERTQ2":
            RBERTQ2_data_preprocessor(parser_arguments['input'], 
                                      parser_arguments['csvoutput'],
                                      parser_arguments['featuresoutput'])
        else: 
            RelationAwareAttention_data_preprocessor(parser_arguments['input'], 
                                                     parser_arguments['csvoutput'])
    if parser_arguments['subparser_name'] == "training":
        if parser_arguments['experiment'] == "RBERTQ2":
            traindata = TextGraphDatasetLoader(parser_arguments['trainpreprocessedfile'])
            evaldata = TextGraphDatasetLoader(parser_arguments['evalpreprocessedfile'])
            evaluation_preds = Training(parser_arguments['experiment']).train_wrapper(traindata, 
                                    parser_arguments['save'], 
                                    parser_arguments['batchsize'],
                                    parser_arguments['epochs'],
                                    evaldata)
        else:
            traindata = TextGraphCandidateDatasetLoader(parser_arguments['trainpreprocessedfile'])
            evaldata = TextGraphCandidateDatasetLoader(parser_arguments['evalpreprocessedfile'])
            evaluation_preds = Training(parser_arguments['experiment']).train_wrapper(traindata, 
                                    parser_arguments['save'], 
                                    parser_arguments['batchsize'],
                                    parser_arguments['epochs'],
                                    evaldata)
    if parser_arguments['subparser_name'] == "ranking":
        # traindata, trainlabels, train_uids = load_preprocessed_data(parser_arguments['trainpreprocessedfile'])
        #evaldata, labels, eval_ids = load_preprocessed_data(parser_arguments['evalpreprocessedfile'])
        evaldata = TextGraphDatasetLoader(parser_arguments['trainpreprocessedfile'])
        saved_model = parser_arguments['model']
        batchsize = parser_arguments['batchsize']
        output_path = parser_arguments['rankingoutput']
        training_obj = Training()
        predictions = training_obj.test(evaldata, saved_model, batchsize)
        rankings = training_obj.ranking(predictions, output_path)
        write_utils.write_textgraph_run_file(rankings, output_path)
        

        #print(loss)
        #print(preds[0])
        