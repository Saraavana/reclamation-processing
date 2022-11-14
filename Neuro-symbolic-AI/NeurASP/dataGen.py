import torch
from torch.utils.data import Dataset
import pandas as pd

import os, sys; 
column_path = os.path.dirname(os.path.realpath('C:/Users/sgopalakrish/Downloads/intellizenz-model-training/Neuro-symbolic-AI/column.py'))
if sys.path.__contains__(column_path)==False:
    sys.path.append(column_path)

import column

def get_weighted_sampler(path):
    y = get_all_target_data(path)
    
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

def get_all_target_data(path): 
    data_df = pd.read_parquet(path) 
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
    def __init__(self, path):
        self.path = path

        # 1. Load the data 
        data_df = pd.read_parquet(path)

        features = column.features_v3 #142 features

        data_df = data_df[features]
        data_df = data_df.fillna(-1) # Fill the Empty NaN values in all the cells with -1

        X = data_df.loc[:,~data_df.columns.isin(['veranst_segment','vg_inkasso'])] #140 features 
        y = data_df['veranst_segment']

        self.X = torch.Tensor(X.values) #dimension: [n, 140]
        self.y = torch.Tensor(y.values) #dimension: [n]
                    
    def __getitem__(self, index):
        return self.X[index], int(self.y[index])
       
    def __len__(self):
        return len(self.y)

train_path = 'C:/Users/sgopalakrish/Downloads/intellizenz-model-training/data/export_training_features_2016_2020_v1.parquet.gzip' 
test_path = 'C:/Users/sgopalakrish/Downloads/intellizenz-model-training/data/export_testing_features_2016_2020_v1.parquet.gzip'

weighted_sampler, class_weights = get_weighted_sampler(train_path)

# Return n batches, where each batch contain batch_size values. Each value has a tensor of features 
# and its target value event(veranst) segment(from 0 to 2)
train_loader = torch.utils.data.DataLoader(Intellizenz(path=train_path), batch_size=64, sampler=weighted_sampler)
test_loader = torch.utils.data.DataLoader(Intellizenz(path=test_path), batch_size=64, shuffle=True)

dataList = []
queryList = []
#train_loader has n batches, each batch contains 64 values
for data_batch, label_batch in train_loader:
    for i, data in enumerate(data_batch):
        x = data_batch[i]
        y = label_batch[i]

        query = ":- not event(t1,{}). ".format(int(y))
        dataList.append({'t1': x})
        queryList.append(query)

    # y = data[1]
    # print('X is :',x[i])
    # print('Target is :',y[i])
    
    # print('X is: ',x)
    # print('Length X is: ',len(x[0]))
    # print('Y is: ',y)
    # print('Length Y is: ',len(y))
    

