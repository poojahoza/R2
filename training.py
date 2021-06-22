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



def training(final_dataset):
    
    config = BertConfig()
    model = RBERTQ1(config=config)
    #print(model)
    print("Started with training")

    trainloader = torch.utils.data.DataLoader(final_dataset, batch_size=16, shuffle=True, num_workers=2)
    #classes = ('0', '1')
    class_weights = torch.Tensor([4.5, 1])
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=class_weights)
    dataiter = iter(trainloader)
    sample_features = dataiter.next()
    labels = ['0','1']
    optimizer = optim.Adam(sample_features, lr=2e-5, )
    
    for epoch in range(1):
    
      running_loss = 0.0
      for i, data in enumerate(trainloader):
        inputs = data
        optimizer.zero_grad()
    
        outputs = model(sample_features[0], sample_features[1], sample_features[2], [0, 1],  sample_features[3], sample_features[4], sample_features[5], sample_features[6], sample_features[7])
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
    
        running_loss += loss.item()
    
        if i % 2000 == 1999:
          print('[%d, %5d] loss : %.3f' % (epoch+1, i+1, running_loss/2000))
          running_loss = 0.0
    
    print('Finished Training')
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="training")
    parser.add_argument("--input")
    parser.add_argument("--output")
    args = parser.parse_args()
    dataset = RBERTQ1_data_preprocessor(args.input, args.output)
    training(dataset)