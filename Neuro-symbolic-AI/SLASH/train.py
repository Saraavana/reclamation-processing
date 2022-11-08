from sklearn.utils import shuffle
from data_generation import Intellizenz, Intellizenz_Data
from pathlib import Path
from rtpt import RTPT

import slash
import torch
import time
import numpy as np
import utils

from einsum_wrapper import EiNet
from network_nn import Net_nn, Simple_nn

from sklearn.metrics import confusion_matrix
import wandb
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

program ='''
tab(t1).
pred(p1).
pred(p2). #--amount

npp(vgsegment(1,T),[0,1,2]) :- tab(T).
event(N,C) :- vgsegment(0,+T,-C), pred(N).

'''

# Query
# :- not event(p1,1).
# Cardinality constraint, it is not the case, that the instance is not an event and it does not belong to class 1

# :- event(p1,2).
# :- event(p1,0).

# Maybe, events with 3 parameters, with id, category, estimated cost of the event. 
# constraint can be, it is not possible, 
# :-not event(eventId=32,cat=1,cost>cost cat 1)

def slash_intellizenz(exp_name, exp_dict):
    # Create RTPT object
    rtpt = RTPT(name_initials=exp_dict['credentials'], experiment_name='SLASH intellizenztype', max_iterations=int(exp_dict['epochs']))

    # Start the RTPT tracking
    rtpt.start()
    
    saveModelPath = './Neuro-symbolic-AI/SLASH/data/'+exp_name+'/slash_models.pt'
    Path("./Neuro-symbolic-AI/SLASH/data/"+exp_name+"/").mkdir(parents=True, exist_ok=True)

    print("Experiment parameters:", exp_dict)

    wandb.init(project="Intellizenz", entity="elsaravana")
    wandb.config = {
        "learning_rate": exp_dict['lr'],
        "epochs": exp_dict['epochs'],
        "batch_size": exp_dict['bs']
    }

    train_path = 'C:/Users/sgopalakrish/Downloads/intellizenz-model-training/data/export_training_features_2016_2020_v1.parquet.gzip' 
    test_path = 'C:/Users/sgopalakrish/Downloads/intellizenz-model-training/data/export_testing_features_2016_2020_v1.parquet.gzip'

    #NETWORKS
    if exp_dict['credentials']=='SNN':   
        #Intellizenztype network
        intellizenz_net = Net_nn(80) # 152 - number of features/columns
        slash_with_nn(intellizenz_net, exp_dict, saveModelPath, rtpt, train_path, test_path)
    else:
        intellizenz_net = Simple_nn(236).model.to(device)
        simple_nn(intellizenz_net, exp_dict, saveModelPath, rtpt, train_path, test_path)

    
# Training the model with SLASH + Neural Network
def slash_with_nn(intellizenz_net, exp_dict, saveModelPath, rtpt, train_path, test_path):
    #trainable params
    num_trainable_params = [sum(p.numel() for p in intellizenz_net.parameters() if p.requires_grad)]
    num_params = [sum(p.numel() for p in intellizenz_net.parameters())]

    
    print("training with {}({}) trainable params and {}({}) params in total".format(np.sum(num_trainable_params),num_trainable_params,np.sum(num_params),num_params)) 
    
    #create the SLASH Program
    nnMapping = {'vgsegment': intellizenz_net}    
    

    #OPTIMIZERS and LEARNING RATE SHEDULING
    optimizers = {'intellizenz': torch.optim.Adam([
                                            {'params':intellizenz_net.parameters()}],
                                            lr=exp_dict['lr'], eps=1e-7)}

    print('-------------------')
    SLASHobj = slash.SLASH(program, nnMapping, optimizers)
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
        intellizenz_net.load_state_dict(saved_model['intellizenz_net'])
      
        #optimizers and shedulers
        optimizers['intellizenz'].load_state_dict(saved_model['resume']['optimizer_intellizenz'])
        start_e = saved_model['resume']['epoch']
       
        #metrics
        train_acc_list = saved_model['train_acc_list']
        test_acc_list = saved_model['test_acc_list']        
        

    # Return n batches, where each batch contain exp_dict['bs'] values. Each value has a tensor of features and its target value event(veranst) segment(from 0 to 2)
    train_data_loader = torch.utils.data.DataLoader(Intellizenz(path=train_path), batch_size=exp_dict['bs'], shuffle=True)

    train_loader = torch.utils.data.DataLoader(Intellizenz_Data(path=train_path), batch_size=exp_dict['bs'], shuffle=True)
    # train_loader = torch.utils.data.DataLoader(Intellizenz_Data(path=train_path), batch_size=exp_dict['bs'], sampler=weighted_sampler)
    test_loader = torch.utils.data.DataLoader(Intellizenz_Data(path=test_path), batch_size=exp_dict['bs'], shuffle=True)
   

    for e in range(start_e, exp_dict['epochs']):
        
        #TRAIN
        print('Epoch {}/{}...'.format(e+1, exp_dict['epochs']))
        time_train= time.time()

        # dataset_loader format ---> {'t1':tensor of features}, rule/query 
        # (rule or query->ex:":- not event(p1,0). ") 
        # Here 0 - veranstaltung(event) segment < 50 euros
        # 1 - veranstaltung(event) segment >50 euros and < 100 euros
        # 2 - veranstaltung(event) segment > 100 euros
        SLASHobj.learn(dataset_loader = train_data_loader, epoch=1, batchSize=exp_dict['bs'],
                              p_num=exp_dict['p_num'], use_em=exp_dict['use_em'])
        
        #TEST
        time_test = time.time()

        # To see gradients of the weights as histograms in the 
        wandb.watch(intellizenz_net)

        #test accuracy
        train_acc, _, = SLASHobj.testNetwork('vgsegment', train_loader, ret_confusion=False)
        test_acc, _, = SLASHobj.testNetwork('vgsegment', test_loader, ret_confusion=False)

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
        torch.save({"intellizenz_net":  intellizenz_net.state_dict(), 
                    "resume": {
                        "optimizer_intellizenz":optimizers['intellizenz'].state_dict(),
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

# Training the model with only simple Neural Network
def simple_nn(intellizenz_net, exp_dict, saveModelPath, rtpt, train_path, test_path):
    #trainable params
    num_trainable_params = [sum(p.numel() for p in intellizenz_net.parameters() if p.requires_grad)]
    num_params = [sum(p.numel() for p in intellizenz_net.parameters())]

    print("training with {}({}) trainable params and {}({}) params in total".format(np.sum(num_trainable_params),num_trainable_params,np.sum(num_params),num_params))

    #OPTIMIZERS and LEARNING RATE SHEDULING
    optimizer = torch.optim.Adam(params= intellizenz_net.parameters(), lr=exp_dict['lr'], eps=1e-7)
    
    #metric lists
    train_acc_list = [] #stores acc for train 
    test_acc_list = []  #and test

    startTime = time.time()
  
    start_e= 0
    if exp_dict['resume']:
        print("resuming experiment")
        saved_model = torch.load(saveModelPath)
        
        #load pytorch models
        intellizenz_net.load_state_dict(saved_model['intellizenz_net'])
      
        #optimizers and shedulers
        optimizer.load_state_dict(saved_model['resume']['optimizer_intellizenz'])
        start_e = saved_model['resume']['epoch']
       
        #metrics
        train_acc_list = saved_model['train_acc_list']
        test_acc_list = saved_model['test_acc_list']        



    # Return n batches, where each batch contain exp_dict['bs'] values. Each value has a tensor of features and its target value event(veranst) segment(from 0 to 2)

    # train_loader = torch.utils.data.DataLoader(Intellizenz_Data(path=train_path), batch_size=exp_dict['bs'], shuffle=True)
    weighted_sampler, class_weights = get_weighted_sampler(train_path)
    train_loader = torch.utils.data.DataLoader(Intellizenz_Data(path=train_path), batch_size=exp_dict['bs'], sampler=weighted_sampler)
    test_loader = torch.utils.data.DataLoader(Intellizenz_Data(path=test_path), batch_size=exp_dict['bs'], shuffle=True)
   
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
    # loss_fn = torch.nn.BCELoss()


    for e in range(start_e, exp_dict['epochs']):
        #TRAIN
        print('Epoch {}/{}...'.format(e+1, exp_dict['epochs']))
        time_train= time.time()


        # dataset_loader format ---> {'t1':tensor of features}, rule/query 
        # (rule or query->ex:":- not event(p1,0). ") 
        # Here 0 - veranstaltung(event) segment < 50 euros
        # 1 - veranstaltung(event) segment >50 euros and < 100 euros
        # 2 - veranstaltung(event) segment > 100 euros
        train_epoch_loss = 0
        train_epoch_acc = 0   

        for data, target in train_loader:
            # forward
            output = intellizenz_net(data.to(device))

            # output_softmax = torch.log_softmax(output, dim = 1)
            # _, output_tags = torch.max(output_softmax, dim = 1) 
            # loss = loss_fn(output_tags, target)

            loss = loss_fn(output, target.to(device))
            train_acc = multi_acc(output, target.to(device))
            

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_epoch_loss += loss.item()
            train_epoch_acc += train_acc.item()

        
        # compute the training loss of the epoch
        train_loss = train_epoch_loss / len(train_loader)
        # train_accuracy = train_epoch_acc / len(train_loader)
        
        
        # To see gradients of the weights as histograms in the 
        wandb.watch(intellizenz_net)

        #TEST
        time_test = time.time()

        #test accuracy
        train_acc, _, = testNetwork(intellizenz_net, train_loader, ret_confusion=False)
        test_acc, _, = testNetwork(intellizenz_net, test_loader, ret_confusion=False)

        print("Test Accuracy:",test_acc)
        print("Train Accuracy:",train_acc)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

        # show
        print('Epoch: {}, Train_loss: {}, Train_accuracy: {:.6f}, Test_accuracy: {:.6f}'.format(e+1,train_loss, train_acc, test_acc))
        
        wandb.log({"train_loss": train_loss, 
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
        torch.save({"intellizenz_net":  intellizenz_net.state_dict(), 
                    "resume": {
                        "optimizer_intellizenz":optimizer.state_dict(),
                        "epoch":e+1
                    },
                    "train_acc_list":train_acc_list,
                    "test_acc_list":test_acc_list,
                    "num_params": num_params,
                    "time": time_array,
                    "exp_dict":exp_dict}, saveModelPath)
        
        # Update the RTPT
        rtpt.step()

def testNetwork(network, testLoader, ret_confusion=False):
        """
        Return a real number in [0,100] denoting accuracy
        @network is the name of the neural network or probabilisitc circuit to check the accuracy. 
        @testLoader is the input and output pairs.
        """
        network.eval()
        # check if total prediction is correct
        correct = 0
        total = 0
        
        #list to collect targets and predictions for confusion matrix
        y_target = []
        y_pred = []

        with torch.no_grad():
            for data, target in testLoader:               
                output = network(data.to(device))
                pred = np.array([int(i[0]<0.5) for i in output.tolist()])
                target = target.numpy()
                correct += (pred.reshape(target.shape) == target).sum()
                total += len(pred)
        accuracy = correct / total
        
        singleAccuracy = 0

        if ret_confusion:
            confusionMatrix = confusion_matrix(np.array(y_target), np.array(y_pred))
            return accuracy, singleAccuracy, confusionMatrix

        return accuracy, singleAccuracy

# Estimate the multi-class accuracy
def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)    
    
    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)
    
    acc = torch.round(acc * 100)
    
    return acc

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

def get_all_target_train_data(path): 
    data_df = pd.read_parquet(path) 
    y = data_df['veranst_segment']
    return y

def get_weighted_sampler(path):
    y_train = get_all_target_train_data(path)
    
    # we obtain a tensor of all target values in the training data
    target_list = []
    for target in y_train.values:
        target_list.append(target)

    target_list = torch.tensor(target_list) 

    # Calculate the weight of each class(veranst_segment=0/1/2) in the training data
    class_count = [i for i in get_class_distribution(y_train).values()]
    class_weights = 1./torch.tensor(class_count, dtype=torch.float) 

    # To address class imbalance: WeightedRandomSampler is used to ensure that each mini batch contains samples from all the classes
    # WeightedRandomSampler expects a weight for each sample. We do that using as follows.
    class_weights_all = class_weights[target_list]

    weighted_sampler = torch.utils.data.WeightedRandomSampler(weights=class_weights_all, 
                                                                num_samples=len(class_weights_all),
                                                                replacement=True)

    return weighted_sampler, class_weights