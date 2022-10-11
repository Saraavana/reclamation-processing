print("start importing...")

import time
import sys

# sys.path.append('C:/Saravana/Projects/SLASH/src')
# sys.path.append('C:/Saravana/Projects/SLASH/src/SLASH')
# sys.path.append('C:/Saravana/Projects/SLASH/src/einsum_wrapper.py')

sys.path.append('C:/Saravana/Projects/Intellizenz/intellizenz-model-training/Neuro-symbolic-AI/SLASH/EinsumNetworksmaster/src/EinsumNetwork')


#from EinsumNetwork import EinsumNetwork, Graph
# from EinsumNetworksmaster.src.EinsumNetwork.EinsumNetwork import *
# from EinsumNetworksmaster.src.EinsumNetwork.ExponentialFamilyArray import NormalArray, BinomialArray, CategoricalArray

#torch, numpy, ...
import torch


import numpy as np
import importlib

#own modules
from data_gen import COVTYPE, FOREST_COVERAGE_TYPE
# from data_gen import test_loader, train_loader, dataList, queryList


from einsum_wrapper import EiNet
from slash import SLASH
import utils
from pathlib import Path

from rtpt import RTPT

#seeds
torch.manual_seed(42)
np.random.seed(42)
print("...done")

# We denote a given variable with + and the query variable with −
# with the query: color_attr(+X, −C) one is asking for P (C|X).
# with query: color_attr(+X, +C) for P (X, C).
# color_attr(−X, −C),we are querying for the prior P (C).
#Forest coverage type - 1 to 7 classes

# program ='''
# tab(t1).
# pred(p1).
# pred(p2).

# npp(covtype(1,T),[1,2,3,4,5,6,7]) :- tab(T).
# forest(N,C) :- covtype(0,+T,-C), pred(N).
# '''

program ='''
tab(t1).
pred(p1).

npp(covtype(1,T),[1,2,3,4,5,6,7]) :- tab(T).
forest(N,C) :- covtype(0,+T,-C), pred(N).

'''

#:- not forest(p1,1) ;  not forest(p2,3).



def slash_covtype(exp_name , exp_dict):
    
    # Create RTPT object
    rtpt = RTPT(name_initials=exp_dict['credentials'], experiment_name='SLASH covtype', max_iterations=int(exp_dict['epochs']))

    # Start the RTPT tracking
    rtpt.start()
    
    

    saveModelPath = 'data/'+exp_name+'/slash_models.pt'
    Path("data/"+exp_name+"/").mkdir(parents=True, exist_ok=True)

    print("Experiment parameters:", exp_dict)


    #setup new SLASH program given the network parameters
    if exp_dict['structure'] == 'poon-domingos':
        exp_dict['depth'] = None
        exp_dict['num_repetitions'] = None
        print("using poon-domingos")

    elif exp_dict['structure'] == 'binary-trees':
        exp_dict['pd_num_pieces'] = None
        print("using binary-trees")

    
    #NETWORKS
        
    #covtype network
    cov_net = EiNet(structure = exp_dict['structure'],
        pd_num_pieces = exp_dict['pd_num_pieces'],
        depth = exp_dict['depth'],
        num_repetitions = exp_dict['num_repetitions'],
        num_var = 54,
        class_count=7,
        K = exp_dict['k'],
        num_sums= exp_dict['num_sums'],
        use_em= exp_dict['use_em'],
        pd_height=9,
        pd_width=6)

  
    
    

    #trainable params
    num_trainable_params = [sum(p.numel() for p in cov_net.parameters() if p.requires_grad)]

    num_params = [sum(p.numel() for p in cov_net.parameters())]

    
    print("training with {}({}) trainable params and {}({}) params in total".format(np.sum(num_trainable_params),num_trainable_params,np.sum(num_params),num_params))
    
     
    
    #create the SLASH Program
    nnMapping = {'covtype': cov_net}    
    

    #OPTIMIZERS and LEARNING RATE SHEDULING

    optimizers = {'cov': torch.optim.Adam([
                                            {'params':cov_net.parameters()}],
                                            lr=exp_dict['lr'], eps=1e-7)}
  
    print('-------------------')
    SLASHobj = SLASH(program, nnMapping, optimizers)
    print('###################')
    

    
    #metric lists
    train_acc_list = [] #stores acc for train 
    test_acc_list = []  #and test

    startTime = time.time()
  
    
    start_e= 0
    if exp_dict['resume']:
        print("resuming experiment")
        saved_model = torch.load(saveModelPath)
        
        #load pytorch models
        cov_net.load_state_dict(saved_model['cov_net'])
      
        #optimizers and shedulers
        optimizers['cov'].load_state_dict(saved_model['resume']['optimizer_cov'])
        start_e = saved_model['resume']['epoch']
       
        #metrics
        train_acc_list = saved_model['train_acc_list']
        test_acc_list = saved_model['test_acc_list']        
        
    train_dataset = COVTYPE(root='./Neuro-symbolic-AI/SLASH/covtype/data/covtype.data' , mode='train')
    # Return 4600 batches, where each batch contain 100 values. 
    # Each value contains two objects 0: {'t1':tensor of featuers for 100 data} and 1: corresponding queries for 100 data
    train_dataset_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=True) 

    # Return 4600 batches, where each batch contain 100 values. Each value has a tensor of features and its target value forest cover type(from 0 to 6)
    train_loader = torch.utils.data.DataLoader(FOREST_COVERAGE_TYPE(root='./Neuro-symbolic-AI/SLASH/covtype/data/covtype.data',mode='train'), batch_size=100, shuffle=True)
    test_loader = torch.utils.data.DataLoader(FOREST_COVERAGE_TYPE(root='./Neuro-symbolic-AI/SLASH/covtype/data/covtype.data',mode='test'), batch_size=100, shuffle=True)
   

    for e in range(start_e, exp_dict['epochs']):
        
        #TRAIN
        print('Epoch {}/{}...'.format(e+1, exp_dict['epochs']))
        time_train= time.time()
        
        
        # SLASHobj.learn(dataList=dataList, queryList=queryList, method='slot', p_num = 1,
        #                 epoch=1, batchSize=exp_dict['bs'], use_em=exp_dict['use_em']) #smPickle='data/stableModels.pickle',

        # SLASHobj.learnnsp(dataList=dataList, obsList=queryList, epoch=1, batchSize=exp_dict['bs'], method='exact')
        
        # train_dataset_loader format ---> {'t1':tensor of features}, rule/query 
        # (rule or query->ex:":- not forest(p1,1). ") | Here 1 means 1st cover type of forest
        # 1 - Spruce/Fir
        # 2 - Lodgepole Pine
        # 3 - Ponderosa Pine
        # 4 - Cottonwood/Willow
        # 5 - Aspen
        # 6 - Douglas-fir
        # 7 - Krummholz
        SLASHobj.learn(dataset_loader = train_dataset_loader, epoch=1, batchSize=exp_dict['bs'],
                              p_num=exp_dict['p_num'], use_em=exp_dict['use_em'])
        
        #TEST
        time_test = time.time()

        #test accuracy
        train_acc, _, = SLASHobj.testNetwork('covtype', train_loader, ret_confusion=False)
        test_acc, _, = SLASHobj.testNetwork('covtype', test_loader, ret_confusion=False)

        print("Test Accuracy:",test_acc)
        print("Train Accuracy:",train_acc)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        
        timestamp_train = utils.time_delta_now(time_train)
        timestamp_test = utils.time_delta_now(time_test)
        timestamp_total =  utils.time_delta_now(startTime)
        
        print('--- train time:  ---', timestamp_train)
        print('--- test time:  ---' , timestamp_test)
        print('--- total time from beginning:  ---', timestamp_total )
        time_array = [timestamp_train, timestamp_test, timestamp_total]
        
        #save the neural network  such that we can use it later
        print('Storing the trained model into {}'.format(saveModelPath))
        torch.save({"cov_net":  cov_net.state_dict(), 
                    "resume": {
                        "optimizer_cov":optimizers['cov'].state_dict(),
                        "epoch":e+1
                    },
                    "train_acc_list":train_acc_list,
                    "test_acc_list":test_acc_list,
                    "num_params": num_params,
                    "time": time_array,
                    "exp_dict":exp_dict,
                    "program":program}, saveModelPath)
        
        # Update the RTPT
        rtpt.step()

        



