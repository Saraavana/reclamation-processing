from dataGen import dataList, queryList, test_loader, train_loader
import time
from network import Net, testNN
from neurasp import NeurASP
import torch
from pathlib import Path

start_time = time.time()

#############################
# NeurASP program
#############################
program ='''
row(t1).

nn(vgsegment(1,T),[0,1,2]) :- row(T).
event(T,C) :- vgsegment(0,T,C).

'''

########
# Define nnMapping and optimizers, initialze NeurASP object
########
m = Net(n_features=140,output_dim=3)
nnMapping = {'vgsegment': m}
#optimizers and learning rate scheduling
optimizer = {'vgsegment': torch.optim.Adam(m.parameters(), lr=0.001)}
NeurASPobj = NeurASP(program, nnMapping, optimizer)



########
# Start training and testing
########

print('Start training for 1 epoch...')
NeurASPobj.learn(dataList=dataList, obsList=queryList, epoch=1, smPickle=None, bar=True)

device = torch.device('cpu')
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

current_time = time.time()
time_array = [current_time, start_time]

#save the neural network  such that we can use it later
saveModelPath = './Neuro-symbolic-AI/NeurASP/data/'+'1_epoch'+'/slash_models.pt'
Path("./Neuro-symbolic-AI/SLASH/data/"+'1_epoch'+"/").mkdir(parents=True, exist_ok=True)

print('Storing the trained model into {}'.format(saveModelPath))
torch.save({"intellizenz_net":  m.state_dict(), 
            "resume": {
                "optimizer_intellizenz":optimizer.state_dict(),
                "epoch":1
            },
            "num_params": m.parameters(),
            "time": time_array}, saveModelPath)

# check testing accuracy
accuracy, singleAccuracy = testNN(model=m, testLoader=test_loader, device=device)
# check training accuracy
accuracyTrain, singleAccuracyTrain = testNN(model=m, testLoader=train_loader, device=device)

print(f'{accuracyTrain:0.2f}\t{accuracy:0.2f}')
print('--- total time from beginning: %s seconds ---' % int(time.time() - start_time))

