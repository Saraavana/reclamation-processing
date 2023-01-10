from torch.utils.data import Dataset
import torch

import os, sys; 
column_path = os.path.dirname(os.path.realpath('C:/Users/sgopalakrish/Downloads/intellizenz-model-training/Neuro-symbolic-AI/column.py'))
if sys.path.__contains__(column_path)==False:
    sys.path.append(column_path)

import column

class Intellizenz(Dataset):
    def __init__(self, df):
        # 1. Load the data 
        data_df = df
        # features = column.features_v5 #143 features
        features = column.features_v7 #77 features

        data_df = data_df[features]

        data_df = data_df.fillna(-1) # Fill the Empty NaN values in all the cells with -1

        # X = data_df.loc[:,~data_df.columns.isin(['veranst_segment','vg_inkasso','tarif_bez'])] # 140 features
        # y = data_df['veranst_segment']
        # tarif = data_df['tarif_bez']

        X = data_df[features] # 77 Features
        y = df['veranst_segment']
        tarif = df['tarif_bez']

        # self.X = torch.Tensor(X.values) #dimension: [n, 140]
        self.X = torch.Tensor(X.values) #dimension: [n, 77]
        self.y = torch.Tensor(y.values) #dimension: [n]
        
        tarif_values = []
        for each in tarif.values:
            tarif_values.append(each)

        self.tarif = tarif_values #dimension: [n]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        # dataTensor = self.X[index].unsqueeze(0) #dataTensor shape would be [1, 140]
        dataTensor = self.X[index] #dataTensor shape would be [140]

        # data = {'t1':dataTensor, 'ta1':self.tarif[index]}
        data = {'t1':dataTensor}
        ta1 = self.tarif[index]
        # query = ":- not event(t1,{}). ".format(int(self.y[index]))
        # query = ":- not event(t1,{}). \ntarif({}).".format(int(self.y[index]),self.tarif[index])
        # query = ":- not event(t1,{},{}). \ntarif({}).".format(ta1,int(self.y[index]),self.tarif[index])
        query = ":- not event({},{}). \ntarif({}).".format(ta1,int(self.y[index]),self.tarif[index])
        return data, query

class Intellizenz_Data(Dataset):
    def __init__(self, df):
        # 1. Load the data 
        data_df = df
        # features = column.features_v5 #143 features
        features = column.features_v7  #77 features

        data_df = data_df[features]

        data_df = data_df.fillna(-1) # Fill the Empty NaN values in all the cells with -1

        # X = data_df.loc[:,~data_df.columns.isin(['veranst_segment','vg_inkasso', 'tarif_bez'])] #140 features 
        # y = data_df['veranst_segment']

        X = data_df[features] # 77 Features
        y = df['veranst_segment']

        self.X = torch.Tensor(X.values) #dimension: [n, 140]
        self.y = torch.Tensor(y.values) #dimension: [n]
                    
    def __getitem__(self, index):
        return self.X[index], int(self.y[index])
       
    def __len__(self):
        return len(self.y)


class Intellizenz_Test(Dataset):
    def __init__(self, df):
        # 1. Load the data 
        data_df = df
        features = column.features_v5 #143 features

        data_df = data_df[features]

        data_df = data_df.fillna(-1) # Fill the Empty NaN values in all the cells with -1

        X = data_df.loc[:,~data_df.columns.isin(['veranst_segment','vg_inkasso','tarif_bez'])] # 140 features
        y = data_df['veranst_segment']
        tarif = data_df['tarif_bez']

        self.X = torch.Tensor(X.values) #dimension: [n, 140]
        self.y = torch.Tensor(y.values) #dimension: [n]
        
        tarif_values = []
        for each in tarif.values:
            tarif_values.append(each)

        self.tarif = tarif_values #dimension: [n]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        # dataTensor = self.X[index].unsqueeze(0) #dataTensor shape would be [1, 140]
        dataTensor = self.X[index] #dataTensor shape would be [140]
        data = {'t1':dataTensor}
        target = int(self.y[index])

        query = "tarif({}).".format(self.tarif[index])
        # query = ":- not event(t1,{}). \ntarif({}).".format(int(self.y[index]),self.tarif[index])

        return data, target, query

