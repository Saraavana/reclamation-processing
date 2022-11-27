from dataGen import dataList, queryList, test_loader, train_loader
import time
from network import Net, testNN
from neurasp import NeurASP
import torch
import numpy as np

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

# program ='''
# row(t1).

# nn(vgsegment(1,T),[0,1,2]) :- row(T).
# event(T,C) :- vgsegment(0,T,C).

# :- event(T,C), tarif(TA), TA=50, C=0 .
# :- event(T,C), tarif(TA), TA=50, C=1 . 

# '''

program ='''
row(t1).

nn(vgsegment(1,T),[0,1,2]) :- row(T).
event(T,C) :- vgsegment(0,T,C).


event(TA,0) :- vgsegment(0,T,C), tarif(TA), TA!=50.
event(TA,1) :- vgsegment(0,T,C), tarif(TA), TA!=50.
'''


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
# Define nnMapping and optimizers, initialze NeurASP object
########
m = Net(n_features=140,output_dim=3)
nnMapping = {'vgsegment': m}
#optimizers and learning rate scheduling
optimizers = {'nasp_intellizenz': torch.optim.Adam([
                                            {'params':m.parameters()}],
                                            lr=0.001)}
NeurASPobj = NeurASP(program, nnMapping, optimizers)
print(optimizers)


########
# Start training and testing
########
print('Start training for 1 epoch...')
NeurASPobj.learn(dataList=dataList, obsList=queryList, epoch=1, smPickle=None, bar=True)

device = torch.device('cpu')
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

current_time = time.time()
time_array = [current_time, start_time]


# check testing accuracy
accuracy, singleAccuracy, y_target, y_pred, probas = testNN(model=m, testLoader=test_loader, device=device)
# check training accuracy
accuracyTrain, singleAccuracyTrain, _, _, _ = testNN(model=m, testLoader=train_loader, device=device)

print(f'{accuracyTrain:0.2f}\t{accuracy:0.2f}')
print('--- total time from beginning: %s seconds ---' % int(time.time() - start_time))

