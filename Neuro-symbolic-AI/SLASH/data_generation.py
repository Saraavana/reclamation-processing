import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import os, sys; 
column_path = os.path.dirname(os.path.realpath('C:/Users/sgopalakrish/Downloads/intellizenz-model-training/Neuro-symbolic-AI/column.py'))
if sys.path.__contains__(column_path)==False:
    sys.path.append(column_path)

import column

class Intellizenz(Dataset):
    def __init__(self, path):
        self.path = path

        # 1. Load the data 
        data_df = pd.read_parquet(path)

        features = column.features_v3 #142 features
        # features = column.features_v4 #238 features

        data_df = data_df[features]

        data_df = data_df.fillna(-1) # Fill the Empty NaN values in all the cells with -1

        X = data_df.loc[:,~data_df.columns.isin(['veranst_segment','vg_inkasso'])] # 152 features
        # X = data_df.loc[:,~data_df.columns.isin(sfdfs)] # 152 features
        y = data_df['veranst_segment']

        # Encode categorical labels
        # l_enc = LabelEncoder()
        # X['vg_datum_year'] = l_enc.fit_transform(X['vg_datum_year'])

        # def fillmissing(df, feature):
        #     df[feature] = df[feature].fillna(0)
                

        # features_missing= X.columns[X.isna().any()]
        # for feature in features_missing:
        #     fillmissing(X, feature= feature)
        
        # X.info()
        # y = l_enc.fit_transform(y)

        self.X = torch.Tensor(X.values) #dimension: [n, 152]
        self.y = torch.Tensor(y.values) #dimension: [n]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        # print('Index is: ',index)
        dataTensor = self.X[index]
        # query = ":- not event(p1,{}). ".format(int(self.y[index]))
        query = ":- not event(t1,{}). ".format(int(self.y[index]))
        return {'t1':dataTensor}, query


class Intellizenz_Data(Dataset):
    def __init__(self, path):
        self.path = path

        # 1. Load the data 
        data_df = pd.read_parquet(path)

        features = column.features_v3 #142 features 
        # features = column.features_v4 #238 features

        data_df = data_df[features]

        data_df = data_df.fillna(-1) # Fill the Empty NaN values in all the cells with -1

        X = data_df.loc[:,~data_df.columns.isin(['veranst_segment','vg_inkasso'])] #236 features 
        y = data_df['veranst_segment']

        # Encode categorical labels
        # l_enc = LabelEncoder()
        # X['vg_datum_year'] = l_enc.fit_transform(X['vg_datum_year'])

        # y = l_enc.fit_transform(y)
        self.X = torch.Tensor(X.values) #dimension: [n, 236]
        self.y = torch.Tensor(y.values) #dimension: [n]
                    
    def __getitem__(self, index):
        return self.X[index], int(self.y[index])
        
       
    def __len__(self):
        return len(self.y)
