import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split

import os, sys; 
parent_directory = '/Users/saravana/Documents/Work/Master-Thesis/reclamation-processing/Neuro-symbolic-AI/SLASH' 
column_path = os.path.dirname(os.path.realpath('/Users/saravana/Documents/Work/Master-Thesis/reclamation-processing/Neuro-symbolic-AI/column.py'))
if sys.path.__contains__(column_path)==False:
    sys.path.append(column_path)

if sys.path.__contains__(parent_directory)==False:
    sys.path.append(parent_directory)

import column
from sklearn.preprocessing import LabelEncoder


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

# Normalize the values in the column between [0,1] using min-max-scaling
def normalize_columns(dataframe, columns):
    # copy the data
    df_min_max_scaled = dataframe.copy()
    
    # apply normalization technique
    for each_column in columns:
        df_min_max_scaled[each_column] = (df_min_max_scaled[each_column] - df_min_max_scaled[each_column].min()) / (df_min_max_scaled[each_column].max() - df_min_max_scaled[each_column].min())    
    
    # Return normalized data
    return df_min_max_scaled

class Intellizenz(Dataset):
    # def __init__(self, path):
    def __init__(self, data_df):
        # features = column.features_v5 #143 features
        # features = column.features_v10 #21 features - with tarif_bez
        features = column.features_v8 #78 features including tarif_bez
        # features = column.features_v2 #140 features # doesn't include tarif_bez
        # features = column.anonymized_features_v1 #140 features # doesn't include tarif_bez

        data_df_normalized = normalize_columns(data_df, features)

        # X = data_df.loc[:,~data_df.columns.isin(['veranst_segment','vg_inkasso','tarif_bez'])] #140 features 
        X = data_df_normalized[features]
        y = data_df_normalized['veranst_segment']
        tarif = data_df['tarif_bez']

        self.X = torch.Tensor(X.values) #dimension: [n, 140] or [n, 78] or [n, 21]
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

# Return n batches, where each batch contain batch_size values. Each value has a tensor of features 
# and its target value event(veranst) segment(from 0 to 2)
# data_path = 'C:/Users/sgopalakrish/Downloads/intellizenz-model-training/data/export_features_2016_2020_v3.parquet.gzip'

data_path = column.data_path_2016_2020_v5 #Leave-one-out-target-encoding features
df = pd.read_parquet(data_path)

# data_path = column.data_path_2016_2020_v3 #one hot encoded features for 30 frequent features
# data_path = column.anony_data_path_2016_2020_v1 #one hot encoded features for 30 frequent features
# df = pd.read_parquet(data_path)

class_frequency = df.groupby('veranst_segment')['veranst_segment'].transform('count')
# df_sampled = df.sample(n=70000, weights=class_frequency, random_state=2)
df_sampled = df.sample(n=300000, weights=class_frequency, random_state=2)

le = LabelEncoder()
df_sampled['tarif_bez'] = le.fit_transform(df_sampled['tarif_bez'])

all_tarifs_le = [e for e in df_sampled['tarif_bez']]

tarif_classes=le.inverse_transform(all_tarifs_le).tolist()
index_of_tarif = tarif_classes.index('U-ST I (MUSIKER) NL')
print('The index is: ',index_of_tarif)
print('The label encoded value is: ',all_tarifs_le[index_of_tarif])

# df_sampled = df_sampled[df_sampled['tarif_bez'] == 56][:2]


df_train, df_test = train_test_split(df_sampled, test_size=0.2, random_state=1)

    
train_weighted_sampler, train_class_weights = get_weighted_sampler(df_train)
test_weighted_sampler, test_class_weights = get_weighted_sampler(df_test)

# Return n batches, where each batch contain batch_size values. Each value has a tensor of features 
# and its target value event(veranst) segment(from 0 to 2)
train_loader = torch.utils.data.DataLoader(Intellizenz(data_df=df_train), batch_size=64, sampler=train_weighted_sampler)

# test_loader = torch.utils.data.DataLoader(Intellizenz(data_df=df_test), batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(Intellizenz(data_df=df_test), batch_size=64, sampler=test_weighted_sampler)

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

            # data = {'t1':x, 'ta1':tarif}
            data = {'t1':x}

            # query = ":- not event(t1,{}). ".format(int(y))
            query = ":- not event(t1,{}). \ntarif({}).".format(int(y),tarif)
            # query = "tarif({}).".format(tarif)
            # query = ":- not event({},{}). \ntarif({}).".format(tarif,int(y),tarif)

            dataList.append(data)
            queryList.append(query)


test_dataList = []
test_queryList = []

#test_loader has n batches, each batch contains 64 values 
# Each data batch shape is [64,140]
# Each label batch shape is [64]
for data_batch, label_batch, tarif_batch in test_loader: 

    for i, data in enumerate(data_batch):
            x = data_batch[i]
            y = label_batch[i]
            tarif = tarif_batch[i]

            data = {'t1':x}

            query = ":- not event(t1,{}). \ntarif({}).".format(int(y),tarif)

            test_dataList.append(data)
            test_queryList.append(query)


# for data_batch, label_batch, tarif_batch in train_loader: 
#     data_dict_batch = []
#     query_dict_batch = []

#     int_label_batch = []
#     for i, _ in enumerate(label_batch):
#         y = label_batch[i]
#         int_label_batch.append(int(y))

#     data = {'t1':data_batch}
#     query = ":- not event(t1,{}). \ntarif({}).".format(int_label_batch,tarif_batch)

#     dataList.append(data)
#     queryList.append(query)
        

# program ='''
# row(t1).
# tarif(ta1).

# :- event(T,C), tarif(TA), TA="U-ST I (MUSIKER) NL", C=0 .
# :- event(T,C), tarif(TA), TA="U-ST I (MUSIKER) NL", C=1 . 

# nn(vgsegment(1,T),[0,1,2]) :- row(T).
# event(T,C) :- vgsegment(0,T,C). '''

    

