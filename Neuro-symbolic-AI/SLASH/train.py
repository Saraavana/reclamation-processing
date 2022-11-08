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
        intellizenz_net = Simple_nn(80).model
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
    loss_fn = torch.nn.BCELoss()

    
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
    train_loader = torch.utils.data.DataLoader(Intellizenz_Data(path=train_path), batch_size=exp_dict['bs'], shuffle=True)
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
        total_loss = 0

        for data, target in train_loader:
            # forward
            output = intellizenz_net(data)
            loss = loss_fn(output, target)

            wandb.log({"loss": loss})

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        # compute the training loss of the epoch
        avg_loss = total_loss / len(train_loader)
        
        # show
        print('Average_train_loss: {}, total_train_loss: {:.6f}'.format(avg_loss, total_loss.data.item()))
        
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

def testNetwork(self, network, testLoader, ret_confusion=False):
        """
        Return a real number in [0,100] denoting accuracy
        @network is the name of the neural network or probabilisitc circuit to check the accuracy. 
        @testLoader is the input and output pairs.
        """
        network.eval()
        # check if total prediction is correct
        correct = 0
        total = 0
        # check if each single prediction is correct
        singleCorrect = 0
        singleTotal = 0
        
        #list to collect targets and predictions for confusion matrix
        y_target = []
        y_pred = []
        with torch.no_grad():

            for data, target in testLoader:               
                output = network(data.to(self.device))
                if len(self.n) != 0 and self.n[network] > 2 :
                    pred = output.argmax(dim=-1, keepdim=True) # get the index of the max log-probability
                    target = target.to(self.device).view_as(pred)
                    
                    correctionMatrix = (target.int() == pred.int()).view(target.shape[0], -1)
                    y_target = np.concatenate( (y_target, target.int().flatten().cpu() ))
                    y_pred = np.concatenate( (y_pred , pred.int().flatten().cpu()) )
                    
                    
                    correct += correctionMatrix.all(1).sum().item()
                    total += target.shape[0]
                    singleCorrect += correctionMatrix.sum().item()
                    singleTotal += target.numel()
                else: 
                    pred = np.array([int(i[0]<0.5) for i in output.tolist()])
                    target = target.numpy()
                    correct += (pred.reshape(target.shape) == target).sum()
                    total += len(pred)
        accuracy = correct / total

        if len(self.n) != 0 and self.n[network] > 2:
            singleAccuracy = singleCorrect / singleTotal
        else:
            singleAccuracy = 0

        if ret_confusion:
            confusionMatrix = confusion_matrix(np.array(y_target), np.array(y_pred))
            return accuracy, singleAccuracy, confusionMatrix

        return accuracy, singleAccuracy