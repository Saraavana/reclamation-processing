from torch.utils.data import Dataset
import torch

import os, sys; 
column_path = os.path.dirname(os.path.relpath('/Users/saravana/Documents/Work/Master-Thesis/reclamation-processing/Neuro-symbolic-AI/column.py'))

if sys.path.__contains__(column_path)==False:
    sys.path.append(column_path)

import column

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
    def __init__(self, df):
        # features = column.features_v5 #143 features
        # features = column.features_v7 #77 features - without tarif
        # features = column.features_v8 #78 features - with tarif
        # features = column.features_v9 #9 features - with tarif
        # features = column.features_v10 #21 features - with tarif

        # features = column.features_v2 #140 features - without tarif_bez
        # features = column.anonymized_features_v1 #140 features - without tarif_bez
        features = column.features_v8 #78 features - with tarif_bez

        # data_df_normalized = normalize_columns(df, features)

        # X = data_df_normalized[features] #140 or 78 or 21 Features
        # y = data_df_normalized['veranst_segment']
        # tarif = df['tarif_bez']

        X = df[features] #140 or 78 or 21 Features
        y = df['veranst_segment']
        tarif = df['tarif_bez']

        self.X = torch.Tensor(X.values) #dimension: [n, 140] or [n, 78] or [n, 21]
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
        query = ":- not event(t1,{}). \ntarif({}).".format(int(self.y[index]),self.tarif[index])
        # query = ":- not event(t1,{},{}). \ntarif({}).".format(ta1,int(self.y[index]),self.tarif[index])
        # query = ":- not event({},{}). \ntarif({}).".format(ta1,int(self.y[index]),self.tarif[index])

        return data, query

class Intellizenz_Data(Dataset):
    def __init__(self, df):
        # features = column.features_v5 #143 features
        # features = column.features_v7  #77 features without tarif
        # features = column.features_v8 #78 features - with tarif
        # features = column.features_v9 #9 features - with tarif
        # features = column.features_v10 #21 features - with tarif

        # features = column.features_v2 #140 features - without tarif_bez
        # features = column.anonymized_features_v1 #140 features - without tarif_bez   
        features = column.features_v8 #78 features - with tarif_bez
        
        # data_df_normalized = normalize_columns(df, features)

        # X = data_df_normalized[features] #140 or 78 or 21 Features
        # y = data_df_normalized['veranst_segment']

        X = df[features] #140 or 78 or 21 Features
        y = df['veranst_segment']

        self.X = torch.Tensor(X.values) #dimension: [n, 140] or [n, 78] or [n, 21]
        self.y = torch.Tensor(y.values) #dimension: [n]
                    
    def __getitem__(self, index):
        return self.X[index], int(self.y[index])
       
    def __len__(self):
        return len(self.y)


# class Intellizenz_Test(Dataset):
#     def __init__(self, df):
#         # 1. Load the data 
#         data_df = df
#         features = column.features_v5 #143 features

#         data_df = data_df[features]

#         data_df = data_df.fillna(-1) # Fill the Empty NaN values in all the cells with -1

#         X = data_df.loc[:,~data_df.columns.isin(['veranst_segment','vg_inkasso','tarif_bez'])] # 140 features
#         y = data_df['veranst_segment']
#         tarif = data_df['tarif_bez']

#         self.X = torch.Tensor(X.values) #dimension: [n, 140]
#         self.y = torch.Tensor(y.values) #dimension: [n]
        
#         tarif_values = []
#         for each in tarif.values:
#             tarif_values.append(each)

#         self.tarif = tarif_values #dimension: [n]

#     def __len__(self):
#         return len(self.y)

#     def __getitem__(self, index):
#         # dataTensor = self.X[index].unsqueeze(0) #dataTensor shape would be [1, 140]
#         dataTensor = self.X[index] #dataTensor shape would be [140]
#         data = {'t1':dataTensor}
#         target = int(self.y[index])

#         query = "tarif({}).".format(self.tarif[index])
#         # query = ":- not event(t1,{}). \ntarif({}).".format(int(self.y[index]),self.tarif[index])

#         return data, target, query

