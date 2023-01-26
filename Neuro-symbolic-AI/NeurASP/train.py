from dataGen import dataList, queryList, test_loader, train_loader
import time
from network import Net, testNN
from neurasp import NeurASP
import torch
# import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd

import os, sys; 
tabnet_path = os.path.dirname(os.path.realpath('C:/Users/sgopalakrish/Downloads/intellizenz-model-training/Neuro-symbolic-AI/SLASH/TabNet/pytorch_tabnet'))
if sys.path.__contains__(tabnet_path)==False:
    sys.path.append(tabnet_path)

column_path = os.path.dirname(os.path.realpath('C:/Users/sgopalakrish/Downloads/intellizenz-model-training/Neuro-symbolic-AI/column.py'))
if sys.path.__contains__(column_path)==False:
    sys.path.append(column_path)

import column
# from SLASH.TabNet.tabnet_nn import TabNetClass
from pathlib import Path

start_time = time.time()

#############################
# NeurASP program
#############################
# program ='''
# row(t1).
# tarif(ta1).

# :- event(T,C), tarif(TA), TA="U-ST I (MUSIKER) NL", C=0 .
# :- event(T,C), tarif(TA), TA="U-ST I (MUSIKER) NL", C=1 . 

# nn(vgsegment(1,T),[0,1,2]) :- row(T).
# event(T,C) :- vgsegment(0,T,C).

# '''

program ='''
row(t1).

nn(vgsegment(1,T),[0,1,2]) :- row(T).
event(T,C) :- vgsegment(0,T,C).

:- event(T,C), tarif(TA), TA=56, C=0.
:- event(T,C), tarif(TA), TA=56, C=1. 

'''

# program ='''
# row(t1).

# nn(vgsegment(1,T),[0,1,2]) :- row(T).
# event(T,C) :- vgsegment(0,T,C).


# event(TA,0) :- vgsegment(0,T,C), tarif(TA), TA!=56.
# event(TA,1) :- vgsegment(0,T,C), tarif(TA), TA!=56.
# '''


# For 50k data - TA("U-ST I (MUSIKER) NL") = 
# For 70k data - TA = 50
# For 100k data - TA - 
# For 200k data - TA - 
# For 300k data - TA - 56
# For 1.7M data - TA - 

# Query Constraint
# :- not event(t1,2).
# tarif(U-K (MUSIKER)).

# % neural rule nn(..
# Integrity constraint is, it is not the case that an event with tarif(ta1) could belong to class 0 or class 1

# get the tarif from row(t1)
# get the state from row(t1)
# tarif in row(t1) is always greater than 2

# tarif(t1, ta)
# :- tarif(T, TA), event(T,C), TA="U-ST I (MUSIKER) NL", C="CLASS2" # Integrity constraint
# event(T,"CLASS 2"):- tarif(T, TA), event(T,C), TA="U-ST I (MUSIKER) NL"

# :- tarif(t1, ta1), event(T,C), ta1="U-ST I (MUSIKER) NL", C=0 .
# :- tarif(t1, ta1), event(T,C), ta1="U-ST I (MUSIKER) NL", C=1 . 

########
# Method to get categorical columns
########
def get_cat_columns_and_dims():
    data_path = column.data_path_2016_2020_v5
    df = pd.read_parquet(data_path)

    class_frequency = df.groupby('veranst_segment')['veranst_segment'].transform('count')
    df_sampled = df.sample(n=300000, weights=class_frequency, random_state=2)

    nunique_clf2 = df_sampled.nunique()
    types_clf2 = df_sampled.dtypes

    categorical_columns = []
    categorical_dims =  {}
    for col in df_sampled.columns:
        if types_clf2[col] == 'object' or nunique_clf2[col] < 200:
            print(col, df_sampled[col].nunique())
            l_enc = LabelEncoder()
            df_sampled[col] = l_enc.fit_transform(df_sampled[col].values)
            categorical_columns.append(col)
            categorical_dims[col] = len(l_enc.classes_)

    return categorical_columns, categorical_dims

########
# Define nnMapping and optimizers, initialze NeurASP object
########
# categorical_columns, categorical_dims = get_cat_columns_and_dims()
# feature_columns = column.features_v10 #21 features - with tarif
# cat_idxs = [ i for i, f in enumerate(feature_columns) if f in categorical_columns]
# cat_dims = [ categorical_dims[f] for i, f in enumerate(feature_columns) if f in categorical_columns]

m = Net(n_features=140,output_dim=3)
# m = Net(n_features=21,output_dim=3)
# m = Net(n_features=78,output_dim=3)
# m = TabNetClass(input_dim=21, output_dim= 3,
#                 n_d=64, n_a=64, n_steps=5,
#                 gamma=1.5, n_independent=2, n_shared=2,
#                 cat_idxs=cat_idxs,
#                 cat_dims=cat_dims,
#                 cat_emb_dim=2,
#                 lambda_sparse=1e-4, momentum=0.3, epsilon=1e-15,
#                 virtual_batch_size=256, 
#                 mask_type='entmax' # sparsemax
#                 )

nnMapping = {'vgsegment': m}
#optimizers and learning rate scheduling
optimizers = {'nasp_intellizenz': torch.optim.Adam([
                                            {'params':m.parameters()}],
                                            lr=0.001)}
NeurASPobj = NeurASP(program, nnMapping, optimizers, gpu=True)
print(optimizers)


########
# Start training and testing
########

#save the neural network  such that we can use it later
# saveModelPath = './Neuro-symbolic-AI/NeurASP/data/'+'1_epoch'+'/slash_models.pt'
# Path("./Neuro-symbolic-AI/SLASH/data/"+'1_epoch'+"/").mkdir(parents=True, exist_ok=True)


print('Start training for 1 epoch...')
# NeurASPobj.learn(dataList=dataList, obsList=queryList, epoch=50, smPickle=None, bar=True)


##Resume training
########
# To resume the training, load the model
########
print("resuming experiment")
saveModelPath = './Neuro-symbolic-AI/NeurASP/data/'+'1_epoch'+'/2HL_MLP_lr_0.001_d300k_140feat_ep20_w_tarif.pt'
saved_model = torch.load(saveModelPath)
print(saveModelPath)
#load pytorch models
m.load_state_dict(saved_model['intellizenz_net'], strict=False)


#optimizers and schedulers
optimizers['nasp_intellizenz'].load_state_dict(saved_model['resume']['optimizer_intellizenz'])
start_e = saved_model['resume']['epoch']

NeurASPobj.learn(dataList=dataList, obsList=queryList, epoch=50, smPickle=None, bar=True, start_e=start_e, saveModelPath=saveModelPath)

# device = torch.device('cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

current_time = time.time()
time_array = [current_time, start_time]

# check testing accuracy
accuracy, singleAccuracy, y_target, y_pred, probas = testNN(model=m, testLoader=test_loader, device=device)
# check training accuracy
accuracyTrain, singleAccuracyTrain, _, _, _ = testNN(model=m, testLoader=train_loader, device=device)

print(f'{accuracyTrain:0.2f}\t{accuracy:0.2f}')
print('--- total time from beginning: %s seconds ---' % int(time.time() - start_time))

