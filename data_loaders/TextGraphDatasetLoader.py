import pandas as pd
import numpy as np
import torch
import json
from torch.utils.data import Dataset


class TextGraphDatasetLoader(Dataset):

    def __init__(self, input_file):
    
        #read csv file and load the variables in the dataframe
        file_data = pd.read_csv(input_file)
        file_data_len = len(file_data)
        print(file_data_len)
        self.x = file_data.iloc[0:file_data_len, 0:5].values
        #print(x)
        self.y = file_data.iloc[0:file_data_len, 5]
        #print(y)
        self.uids = file_data.iloc[0:file_data_len, 7]
        #print(uids)
          
        #print(x.tolist())
        #print(len(x[0]))
        #print(json.loads(x[0][0]))
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        #print("in get item")
        #print(idx)
        #print(self.x[idx])
        index_tokens_tensor = torch.tensor(json.loads(self.x[idx][0]))
        # #print(index_tokens_tensor.shape)
        segments_tensor = torch.tensor(json.loads(self.x[idx][1]))
        # #print(len(json.loads(self.x[idx][1])))
        # #print(segments_tensor.shape)
        attn_mask_tensor = torch.tensor(json.loads(self.x[idx][2]))
        # #print(attn_mask_tensor.shape)
        ent1_mask_tensor = torch.tensor(json.loads(self.x[idx][3]))
        # #print(ent1_mask_tensor.shape)
        ent2_mask_tensor = torch.tensor(json.loads(self.x[idx][4]))
        
        self.x_train = []
        self.x_train.append(index_tokens_tensor)
        self.x_train.append(segments_tensor)
        self.x_train.append(attn_mask_tensor)
        self.x_train.append(ent1_mask_tensor)
        self.x_train.append(ent2_mask_tensor)
        self.y_train = torch.tensor(json.loads(self.y[idx])[0])
        
        self.z_train = self.uids[idx]
        # print(self.z_train)
        return self.x_train, self.y_train, self.z_train

