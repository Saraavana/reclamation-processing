import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

import os, sys; 
column_path = os.path.dirname(os.path.realpath('C:/Users/sgopalakrish/Downloads/intellizenz-model-training/Neuro-symbolic-AI/column.py'))
if sys.path.__contains__(column_path)==False:
    sys.path.append(column_path)

import column

def get_weighted_sampler(data_df):
    # y = get_all_target_data(path)
    y = get_all_target_data(data_df)
    
    # we obtain a tensor of all target values in the training data
    target_list = []
    for target in y.values:
        target_list.append(target)

    target_list = torch.tensor(target_list) 

    # Calculate the weight of each class(veranst_segment=0/1/2) in the training data
    class_count = [i for i in get_class_distribution(y).values()]
    class_weights = 1./torch.tensor(class_count, dtype=torch.float) 

    # To address class imbalance: WeightedRandomSampler is used to ensure that each mini batch contains samples from all the classes
    # WeightedRandomSampler expects a weight for each sample. We do that using as follows.
    class_weights_all = class_weights[target_list]

    weighted_sampler = torch.utils.data.WeightedRandomSampler(weights=class_weights_all, 
                                                                num_samples=len(class_weights_all),
                                                                replacement=True)
    return weighted_sampler, class_weights

def get_all_target_data(data_df): 
    # data_df = pd.read_parquet(path) 
    y = data_df['veranst_segment']
    return y

# Estimate the distribution(frequency/count) of each class(veranst_segment-0,1,2)
def get_class_distribution(obj):
    count_dict = {
        "0": 0,
        "1": 0,
        "2": 0,
    }
    
    for i in obj:
        if i == 0: 
            count_dict['0'] += 1
        elif i == 1: 
            count_dict['1'] += 1
        elif i == 2: 
            count_dict['2'] += 1            
        else:
            print("Check classes.")
            
    return count_dict

class Intellizenz(Dataset):
    # def __init__(self, path):
    def __init__(self, data_df):
        # self.path = path

        # # 1. Load the data 
        # data_df = pd.read_parquet(path)

        features = column.features_v5 #143 features

        data_df = data_df[features]
        data_df = data_df.fillna(-1) # Fill the Empty NaN values in all the cells with -1

        X = data_df.loc[:,~data_df.columns.isin(['veranst_segment','vg_inkasso','tarif_bez'])] #140 features 
        y = data_df['veranst_segment']
        tarif = data_df['tarif_bez']

        self.X = torch.Tensor(X.values) #dimension: [n, 140]
        self.y = torch.Tensor(y.values) #dimension: [n]

        tarif_values = []
        for each in tarif.values:
            tarif_values.append(each)

        self.tarif = tarif_values #dimension: [n]
                    
    def __getitem__(self, index):
        x_value = self.X[index].unsqueeze(0) #x tensor shape would be [1, 140]
        return x_value, int(self.y[index]), self.tarif[index]
       
    def __len__(self):
        return len(self.y)

class Intellizenz_Test(Dataset):
    # def __init__(self, path):
    def __init__(self, data_df):
        # self.path = path

        # # 1. Load the data 
        # data_df = pd.read_parquet(path)

        features = column.features_v5 #143 features

        data_df = data_df[features]
        data_df = data_df.fillna(-1) # Fill the Empty NaN values in all the cells with -1

        X = data_df.loc[:,~data_df.columns.isin(['veranst_segment','vg_inkasso','tarif_bez'])] #140 features 
        y = data_df['veranst_segment']
        tarif = data_df['tarif_bez']

        self.X = torch.Tensor(X.values) #dimension: [n, 140]
        self.y = torch.Tensor(y.values) #dimension: [n]

        tarif_values = []
        for each in tarif.values:
            tarif_values.append(each)

        self.tarif = tarif_values #dimension: [n]
                    
    def __getitem__(self, index):
        x_value = self.X[index].unsqueeze(0) #x tensor shape would be [1, 140]
        return x_value, int(self.y[index]), self.tarif[index]
       
    def __len__(self):
        return len(self.y)


# train_path = 'C:/Users/sgopalakrish/Downloads/intellizenz-model-training/data/export_training_features_2016_2020_v1.parquet.gzip' 
# test_path = 'C:/Users/sgopalakrish/Downloads/intellizenz-model-training/data/export_testing_features_2016_2020_v1.parquet.gzip'

# weighted_sampler, class_weights = get_weighted_sampler(train_path)

# # Return n batches, where each batch contain batch_size values. Each value has a tensor of features 
# # and its target value event(veranst) segment(from 0 to 2)
# train_loader = torch.utils.data.DataLoader(Intellizenz(path=train_path,train=True), batch_size=64, sampler=weighted_sampler)
# test_loader = torch.utils.data.DataLoader(Intellizenz(path=test_path, train=False), batch_size=64, shuffle=True)


data_path = 'C:/Users/sgopalakrish/Downloads/intellizenz-model-training/data/export_features_2016_2020_v3.parquet.gzip'
df = pd.read_parquet(data_path)

# u_st_nl_df = df.loc[df['tarif_bez']=='U-ST I (MUSIKER) NL']

train_df, test_df = train_test_split(df, test_size=0.2, random_state=1)
weighted_sampler, class_weights = get_weighted_sampler(train_df)

# Return n batches, where each batch contain batch_size values. Each value has a tensor of features 
# and its target value event(veranst) segment(from 0 to 2)
train_loader = torch.utils.data.DataLoader(Intellizenz(data_df=train_df), batch_size=64, sampler=weighted_sampler)

# only randomly take 30000 data
np.random.seed(1) # fix the random seed for reproducibility
# train_loader = torch.utils.data.Subset(train_loader, np.random.choice(len(train_loader), 21778, replace=False))
test_loader = torch.utils.data.DataLoader(Intellizenz(data_df=test_df), batch_size=64, shuffle=True)

# tarif = 'U-ST I (MUSIKER) NL' in trainloader(from 1.3M data) - 18094 - less frequency than actual cause of weighted_sampler
# tarif = 'U-ST I (MUSIKER) NL' in testloader - 5662

dataList = []
queryList = []
#train_loader has n batches, each batch contains 64 values 
# Each data batch shape is [64,140]
# Each label batch shape is [64]
for data_batch, label_batch, tarif_batch in train_loader: 
    for i, data in enumerate(data_batch):
        # if i==0:
            x = data_batch[i]
            y = label_batch[i]
            tarif = tarif_batch[i]

            query = ":- not event(t1,{}). ".format(int(y))
            dataList.append({'t1': x, 'ta1':tarif})
            queryList.append(query)


program ='''
row(t1).
tarif(ta1).

:- event(T,C), ta1="U-ST I (MUSIKER) NL", C=0 .
:- event(T,C), ta1="U-ST I (MUSIKER) NL", C=1 . 

nn(vgsegment(1,T),[0,1,2]) :- row(T).
event(T,C) :- vgsegment(0,T,C). '''

    

