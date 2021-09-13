# -*- coding: utf-8 -*-

import pandas as pd
from torch.utils.data import Dataset


class TextGraphCandidateDatasetLoader(Dataset):

    def __init__(self, input_file: str):
    
        #read csv file and load the variables in the dataframe
        file_data = pd.read_csv(input_file)
        file_data_len = len(file_data)
        print(file_data_len)
        
        self.x = file_data.iloc[0:file_data_len, 1:3].values
        self.y = file_data.iloc[0:file_data_len, 3]
        self.uids = file_data.iloc[0:file_data_len, 4]
          
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):

        query_text = self.x[idx][0]
        explanation_text = self.x[idx][1]
        
        self.x_train = []
        self.x_train.append(query_text)
        self.x_train.append(explanation_text)
        
        self.y_train = self.y[idx]
        
        self.z_train = self.uids[idx]
        
        return self.x_train, self.y_train, self.z_train

