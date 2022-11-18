from pathlib import Path
from sklearn.model_selection import train_test_split
from tabnet_nn import TabNetClass
from dataGen import *

import os, sys; 
parent_directory = 'C:/Users/sgopalakrish/Downloads/intellizenz-model-training/Neuro-symbolic-AI/SLASH' 
column_path = os.path.dirname(os.path.realpath('C:/Users/sgopalakrish/Downloads/intellizenz-model-training/Neuro-symbolic-AI/column.py'))
if sys.path.__contains__(column_path)==False:
    sys.path.append(column_path)

if sys.path.__contains__(parent_directory)==False:
    sys.path.append(parent_directory)

from SLASH import *
import slash
import utils

import wandb
import torch
import pandas as pd
import numpy as np
import time
import column
from sklearn.preprocessing import LabelEncoder

#############################
# SLASH program
#############################
program ='''
row(t1).

npp(tabnet_vgsegment(1,T),[0,1,2]) :- row(T). # [16,384, 1, 140]
event(T,C) :- tabnet_vgsegment(0,+T,-C).

:- event(T,C), tarif(TA), TA=50, C=0 .
:- event(T,C), tarif(TA), TA=50, C=1 . 

'''

# :-not event(t1,2).
# tarif(34).

def slash_tabnet(exp_name, exp_dict):
    saveModelPath = './Neuro-symbolic-AI/SLASH/TabNet/data/'+exp_name+'/slash_tabnet_model_1.pt'
    Path("./Neuro-symbolic-AI/SLASH/TabNet/data/"+exp_name+"/").mkdir(parents=True, exist_ok=True)

    print("Experiment parameters:", exp_dict)

    wandb.init(project="Intellizenz", entity="elsaravana")
    wandb.config = {
        "learning_rate": exp_dict['lr'],
        "epochs": exp_dict['epochs'],
        "batch_size": exp_dict['bs']
    }

    data_path = column.data_path_2016_2020_v3
    df = pd.read_parquet(data_path)

    class_frequency = df.groupby('veranst_segment')['veranst_segment'].transform('count')
    df_sampled = df.sample(n=70000, weights=class_frequency, random_state=2)

    le = LabelEncoder()
    df_sampled['tarif_bez'] = le.fit_transform(df_sampled['tarif_bez'])

    all_tarifs_le = [e for e in df_sampled['tarif_bez']]

    tarif_classes=le.inverse_transform(all_tarifs_le).tolist()
    index_of_tarif = tarif_classes.index('U-ST I (MUSIKER) NL')
    print('The index is: ',index_of_tarif)
    print('The label encoded value is: ',all_tarifs_le[index_of_tarif])


    train_df, test_df = train_test_split(df_sampled, test_size=0.2, random_state=1)

    #NETWORKS
    if exp_dict['credentials']=='STN':
        model = TabNetClass(input_dim=140, output_dim= 3,
                            n_d=64, n_a=64, n_steps=5,
                            gamma=1.5, n_independent=2, n_shared=2,
                            cat_emb_dim=1,
                            lambda_sparse=1e-4, momentum=0.3,
                            epsilon=1e-15, virtual_batch_size=exp_dict['bs']/8
                            )
        slash_with_tabnet(model, exp_dict, saveModelPath, train_df, test_df)
    else:
        print('##########################')


def slash_with_tabnet(model, exp_dict, saveModelPath, train_df, test_df):
    #trainable params
    num_trainable_params = [sum(p.numel() for p in model.parameters() if p.requires_grad)]
    num_params = [sum(p.numel() for p in model.parameters())]

    
    print("training with {}({}) trainable params and {}({}) params in total".format(np.sum(num_trainable_params),num_trainable_params,np.sum(num_params),num_params)) 
    
    ########
    # Define nnMapping and optimizers, initialze SLASH object
    ########
    nnMapping = {'tabnet_vgsegment': model}
    #optimizers and learning rate scheduling
    optimizers = {'tabnet_vgsegment': torch.optim.Adam([
                                            {'params':model.parameters()}],
                                            lr=exp_dict['lr'], eps=1e-7)}
    SLASHobj = slash.SLASH(program, nnMapping, optimizers)

    #metric lists
    train_acc_list = [] #stores acc for train 
    test_acc_list = []  #and test

    weighted_sampler, class_weights = utils.get_weighted_sampler(train_df)
    test_weighted_sampler, test_class_weights = utils.get_weighted_sampler(test_df)

    startTime = time.time()

    # Return n batches, where each batch contain exp_dict['bs'] values. Each value has a tensor of features and its target value event(veranst) segment(from 0 to 2)
    train_data_loader = torch.utils.data.DataLoader(Intellizenz(df=train_df), batch_size=exp_dict['bs'], sampler=weighted_sampler)
    # train_data_loader = torch.utils.data.DataLoader(Intellizenz(df=train_df), batch_size=2, sampler=weighted_sampler)

    train_loader = torch.utils.data.DataLoader(Intellizenz_Data(df=train_df), batch_size=exp_dict['bs'], sampler=weighted_sampler)
    test_loader = torch.utils.data.DataLoader(Intellizenz_Data(df=test_df), batch_size=exp_dict['bs'], sampler=test_weighted_sampler)
    
    start_e= 0
    if exp_dict['resume']:
        print("resuming experiment")
        saved_model = torch.load(saveModelPath)
        
        #load pytorch models
        model.load_state_dict(saved_model['slash_tabnet'])
      
        #optimizers and shedulers
        optimizers['tabnet_vgsegment'].load_state_dict(saved_model['resume']['optimizer_intellizenz'])
        start_e = saved_model['resume']['epoch']
       
        #metrics
        train_acc_list = saved_model['train_acc_list']
        test_acc_list = saved_model['test_acc_list']  

    for e in range(start_e, exp_dict['epochs']):        
        #TRAIN
        print('Epoch {}/{}...'.format(e+1, exp_dict['epochs']))
        time_train= time.time()
        total_loss = SLASHobj.learn(dataset_loader = train_data_loader, epoch=1, batchSize=exp_dict['bs'],
                              p_num=exp_dict['p_num'], use_em=exp_dict['use_em'])
        
        #TEST
        time_test = time.time()

        # To see gradients of the weights as histograms in the 
        wandb.watch(model)

        #test accuracy
        train_acc, _, _, _, _ = SLASHobj.testNetwork('tabnet_vgsegment', train_loader, ret_confusion=False)
        test_acc, _, preds, targets, probas = SLASHobj.testNetwork('tabnet_vgsegment', test_loader, ret_confusion=False)

        print("Test Accuracy:",test_acc)
        print("Train Accuracy:",train_acc)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

        # 3d array to 2d array
        flatten_list_two_dim = lambda y:[x for a in y for x in a] if type(y) is list else [y]
        probas = flatten_list_two_dim(probas)

        wandb.log({"conf_mat" : wandb.plot.confusion_matrix(probs=None,
                            preds=preds, y_true=targets,
                            class_names=[0, 1, 2])})
        wandb.log({"pr" : wandb.plot.pr_curve(y_true=targets, y_probas=probas,
                     labels=['Segment 0-50€', 'Segment 50-100€', 'Segment >100€'], classes_to_plot=[0, 1, 2])})
        wandb.log({"roc" : wandb.plot.roc_curve(y_true=targets, y_probas=probas,
                        labels=['Segment 0-50€', 'Segment 50-100€', 'Segment >100€'], classes_to_plot=[0, 1, 2])})
        
        wandb.log({"train_loss": total_loss, 
                    "train_accuracy": train_acc,
                    "test_accuracy": test_acc})
        
        timestamp_train = utils.time_delta_now(time_train)
        timestamp_test = utils.time_delta_now(time_test)
        timestamp_total =  utils.time_delta_now(startTime)
        
        print('--- train time:  ---', timestamp_train)
        print('--- test time:  ---' , timestamp_test)
        print('--- total time from beginning:  ---', timestamp_total )
        time_array = [timestamp_train, timestamp_test, timestamp_total]
        
        #save the neural network  such that we can use it later
        print('Storing the trained model into {}'.format(saveModelPath))
        torch.save({"intellizenz_net":  model.state_dict(), 
                    "resume": {
                        "optimizer_intellizenz":optimizers['tabnet_vgsegment'].state_dict(),
                        "epoch":e+1
                    },
                    "train_acc_list":train_acc_list,
                    "test_acc_list":test_acc_list,
                    "num_params": num_params,
                    "time": time_array,
                    "exp_dict":exp_dict,
                    "program":program}, saveModelPath)

