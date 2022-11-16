from dataGen import dataList, queryList, test_loader, train_loader
import time
from network import Net, testNN
from neurasp import NeurASP
import torch
from pathlib import Path
import wandb
import numpy as np

start_time = time.time()

#############################
# NeurASP program
#############################
program ='''
row(t1).
tarif(ta1).

:- event(T,C), tarif(TA), TA="U-ST I (MUSIKER) NL", C=0 .
:- event(T,C), tarif(TA), TA="U-ST I (MUSIKER) NL", C=1 . 

nn(vgsegment(1,T),[0,1,2]) :- row(T).
event(T,C) :- vgsegment(0,T,C).

'''
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
# Define nnMapping and optimizers, initialze NeurASP object
########
m = Net(n_features=140,output_dim=3)
nnMapping = {'vgsegment': m}
#optimizers and learning rate scheduling
optimizers = {'intellizenz': torch.optim.Adam([
                                            {'params':m.parameters()}],
                                            lr=0.001)}
# optimizer = {'vgsegment': torch.optim.Adam(([
                                            # {'params':m.parameters()}]), lr=0.001)}
NeurASPobj = NeurASP(program, nnMapping, optimizers)
print(optimizers)


########
# Start training and testing
########

wandb.init(project="Intellizenz", entity="elsaravana")
wandb.config = {
        "learning_rate": 0.001,
        "epochs": 1
}

print('Start training for 1 epoch...')
NeurASPobj.learn(dataList=dataList, obsList=queryList, epoch=1, smPickle=None, bar=True)

device = torch.device('cpu')
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

current_time = time.time()
time_array = [current_time, start_time]

#save the neural network  such that we can use it later
saveModelPath = './Neuro-symbolic-AI/NeurASP/data/'+'1_epoch'+'/slash_models.pt'
Path("./Neuro-symbolic-AI/SLASH/data/"+'1_epoch'+"/").mkdir(parents=True, exist_ok=True)

# print('Storing the trained model into {}'.format(saveModelPath))
# torch.save({"intellizenz_net":  m.state_dict(), 
#             "resume": {
#                 # "optimizer_intellizenz":optimizers.state_dict(),
#                 "epoch":1
#             },
#             "num_params": m.parameters(),
#             "time": time_array}, saveModelPath)

# check testing accuracy
accuracy, singleAccuracy, y_target, y_pred, probas = testNN(model=m, testLoader=test_loader, device=device)
# check training accuracy
accuracyTrain, singleAccuracyTrain, _, _, _ = testNN(model=m, testLoader=train_loader, device=device)
    
probas = [x for sublist in probas for x in sublist] # probas dim-(n_samples, n_classes)

wandb.log({"train_accuracy": accuracyTrain,
            "test_accuracy": accuracy})

            
wandb.log({"conf_mat" : wandb.plot.confusion_matrix(probs=None,
                preds=y_pred, y_true=y_target,
                class_names=[0, 1, 2])})
wandb.log({"pr" : wandb.plot.pr_curve(y_true=y_target, y_probas=probas,
                labels=['Segment 0-50€', 'Segment 50-100€', 'Segment >100€'], classes_to_plot=[0, 1, 2])})
wandb.log({"roc" : wandb.plot.roc_curve(y_true=y_target, y_probas=probas,
                labels=['Segment 0-50€', 'Segment 50-100€', 'Segment >100€'], classes_to_plot=[0, 1, 2])})

print(f'{accuracyTrain:0.2f}\t{accuracy:0.2f}')
print('--- total time from beginning: %s seconds ---' % int(time.time() - start_time))

