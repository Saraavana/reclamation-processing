# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 10:30:37 2022

@author: yanru
"""

import numpy as np
import torch
from torch.autograd import Variable
from sklearn.metrics import accuracy_score
from Models_node import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define Node class
class Node():
    def __init__(self, nodei):
       self.idx = nodei


# Assign training data to a node
def Givetraintonode(Nodes, pronodenum, datanums):
    Nodes[pronodenum].trainidx = datanums
    return Nodes


# Assign validation data to a node
def Givevaltonode(Nodes, pronodenum, datanums):
    Nodes[pronodenum].testidx = datanums
    return Nodes


# Train a NSTSC model given training data
def Train_model(Xtrain_raw, Xval_raw, ytrain_raw, yval_raw, \
                epochs = 100, normalize_timeseries = True, lr = 0.1):
    classnum = int(np.max(ytrain_raw) + 1)
    Tree = Build_tree(Xtrain_raw, Xval_raw, ytrain_raw, yval_raw, epochs, classnum, \
                   learnrate = lr, savepath = 'C:/Saravana/Projects/Intellizenz/intellizenz-model-training/Neuro-symbolic-AI/NSTSC/Codes/utils/')
    Tree = Prune_tree(Tree, Xval_raw, yval_raw)
    
    return Tree


# Construct a tree from node phase classifiers
def Build_tree(Xtrain, Xval, ytrain_raw, yval_raw, Epoch, classnum, \
               learnrate, savepath = "./Neuro-symbolic-AI/NSTSC/Codes/utils/"):
    Tree = {}
    pronodenum = 0
    maxnodenum = 0
    Modelnum = 7
    bstaccu = 0
    Tree[maxnodenum] = Node(maxnodenum)
    Tree[pronodenum].stoptrain = False
    Tree = Givetraintonode(Tree, pronodenum, list(range(len(ytrain_raw))))
    Tree = Givevaltonode(Tree, pronodenum, list(range(len(yval_raw))))
    while pronodenum <= maxnodenum:

        print('The pronodenum: ',pronodenum)
        print('The maxnodenum: ',maxnodenum)
        # Xtrain = Xtrain.to(device)
        # ytrain_raw = ytrain_raw.to(device)
        # Xval = Xval.to(device)
        # yval_raw = yval_raw.to(device)
        if not Tree[pronodenum].stoptrain:
            Tree, trueidx, falseidx, trueidxt,\
            falseidxt, = Trainnode(Tree, pronodenum, Epoch, learnrate,\
            Xtrain, ytrain_raw, Modelnum, savepath, classnum,\
                Xval, yval_raw)
        
            if maxnodenum < 128:
                if len(Tree[pronodenum].trueidx) > 0:
                    Tree, maxnodenum = Updateleftchd(Tree, pronodenum,\
                    maxnodenum, Xtrain, ytrain_raw, classnum,\
                        Xval, yval_raw)
                if len(Tree[pronodenum].falseidx) > 0:
                    Tree, maxnodenum = Updaterigtchd(Tree, pronodenum,\
                    maxnodenum, Xtrain, ytrain_raw, classnum,\
                        Xval, yval_raw)
    
        pronodenum += 1


    return Tree


# Train a node phase classifier
def Trainnode(Nodes, pronum, Epoch, lrt, X, y, Mdlnum, mdlpath, clsnum, Xt, yt):
    trainidx = Nodes[pronum].trainidx    
    Xori = X[trainidx,:]
    yori = y[trainidx]
    testidx = Nodes[pronum].testidx
    Xorit = Xt[testidx,:]
    yorit = yt[testidx]
    yoricount = County(yori, clsnum)
    yoricountt = County(yorit, clsnum)
    curclasses = np.where(yoricount!=0)[0]
    Nodes[pronum].ycount = yoricount
    Nodes[pronum].predcls = yoricountt.argmax()
    yecds = Ecdlabel(yori, curclasses)
    yecdst = Ecdlabel(yorit, curclasses)
    yori = np.array(yori)
    yori = torch.LongTensor(yori)
    yorit = torch.LongTensor(yorit)
    N, T = len(yori), int(Xori.shape[1]/3)
    ginibest = 10
    Xori = torch.Tensor(Xori)
    Xorit = torch.Tensor(Xorit)
    batch_size = N // 20
    if batch_size <= 1:
        batch_size = N
        
    for mdlnum in range(1, Mdlnum):
        tlnns = {}
        optimizers = {}
        X_rns = {}
        Losses = {}
        for i in curclasses:
            tlnns[i] = eval('TL_NN' + str(mdlnum) + '(T)')
            optimizers[i] = torch.optim.AdamW(tlnns[i].parameters(), lr = lrt)
        
        ginisall = []
        
        print('Model number :{} and Epochs :{}'.format(mdlnum,Epoch))

        for epoch in range(Epoch):        
            for d_i in range(N//batch_size + 1):
                rand_idx = np.array(range(d_i*batch_size, min((d_i+1)*batch_size,\
                                N)))           
                for Ci in curclasses:
                    ytrain = np.array(yecds[Ci])
                    ytest = np.array(yecdst[Ci])
                    IR = sum(ytrain==1)/sum(ytrain==0) 
                    ytrain = torch.LongTensor(ytrain)
                    ytest = torch.LongTensor(ytest)
                    X_batch = Variable(torch.Tensor(Xori[rand_idx,:]))
                    y_batch = ytrain[rand_idx]
                    w_batch = IR * (1-y_batch) 
                    w_batch[w_batch==0] = 1
                    X_rns[Ci] = tlnns[Ci](X_batch[:,:T], X_batch[:,T:2*T],\
                                          X_batch[:,2*T:])
                    # print(X_rns[Ci].device)
                    X_rns[Ci] = X_rns[Ci].to(device)
                    y_batch = y_batch.to(device)
                    w_batch = y_batch.to(device)
                    # print(y_batch.device)
                    # print(w_batch.device)
                    Losses[Ci] =  torch.sum(w_batch * (-y_batch * \
                                  torch.log(X_rns[Ci] + 1e-9) - (1-y_batch) * \
                                  torch.log(1-X_rns[Ci] + 1e-9)))
                    
                    optimizers[Ci].zero_grad()
                    Losses[Ci].backward()
                    optimizers[Ci].step()
                
                if d_i % 10 == 0:
                    giniscores = torch.Tensor(Cptginisplit(tlnns, Xorit, yorit,\
                                                           T, clsnum))
                    ginisminnum = int(giniscores.argmin().numpy())
                    ginismin = giniscores.min()
                    ginisall.append(ginismin)
                    if ginismin < ginibest:
                        torch.save(tlnns[curclasses[ginisminnum]], mdlpath + 'bestmodel.pkl')
                        # Nodes[pronum].predcls = ginisminnum
                        Nodes[pronum].ginis = ginismin
                        ginibest = ginismin
                        Nodes[pronum].bstmdlclass = curclasses[ginisminnum]
                    
                    
    Nodes[pronum].bestmodel = torch.load(mdlpath + 'bestmodel.pkl')
                 
    Xpred, accu, trueidx, falseidx = Cpt_Accuracy(Nodes[pronum].bestmodel,\
                                Xori, yecds[Nodes[pronum].bstmdlclass], T)
    Xpredt, accut, trueidxt, falseidxt = Cpt_Accuracy(Nodes[pronum].bestmodel,\
                                Xorit, yecdst[Nodes[pronum].bstmdlclass], T)
        
    
    Nodes[pronum].trueidx = np.array(Nodes[pronum].trainidx)[trueidx]
    Nodes[pronum].falseidx = np.array(Nodes[pronum].trainidx)[falseidx]
    Nodes[pronum].trueidxt = np.array(Nodes[pronum].testidx)[trueidxt]
    Nodes[pronum].falseidxt = np.array(Nodes[pronum].testidx)[falseidxt]
    return Nodes, trueidx, falseidx, trueidxt, falseidxt    


# Expand left child node
def Updateleftchd(Nodes, pronum, maxnum, Xori, yori, clsnum, Xorit, yorit):
    Leftidx = Nodes[pronum].trueidx
    Leftidxt = Nodes[pronum].trueidxt
    yleft = yori[Leftidx]
    ylgini = Cptgininode(yleft, clsnum)
    yleftt = yorit[Leftidxt]
    ylginit = Cptgininode(yleftt, clsnum)
    maxnum += 1
    Nodes[maxnum] = Node(maxnum)
    Nodes = Givetraintonode(Nodes, maxnum, Leftidx)
    Nodes = Givevaltonode(Nodes, maxnum, Leftidxt)
    ylcount = County(yleft, clsnum)
    ylcountt = County(yleftt, clsnum)
    Nodes[maxnum].ycount = ylcount
    Nodes[maxnum].ycountt = ylcountt
    Nodes[maxnum].predcls = ylcountt.argmax()
    Nodes[maxnum].ginis = ylgini
    Nodes[maxnum].ginist = ylginit
    
    if ylginit == 0 or ylgini == 0:
        Nodes[maxnum].stoptrain = True
    else:
        Nodes[maxnum].stoptrain = False
    Nodes[pronum].leftchd = maxnum
    Nodes[maxnum].prntnb = pronum
    Nodes[maxnum].childtype = 'leftchild'
    
    return Nodes, maxnum
    

# Expand right child node
def Updaterigtchd(Nodes, pronum, maxnum, Xori, yori, clsnum, Xorit, yorit):
    Rightidx = Nodes[pronum].falseidx
    Rightidxt = Nodes[pronum].falseidxt
    yright = yori[Rightidx]
    yrgini = Cptgininode(yright, clsnum)
    yrightt = yorit[Rightidxt]
    yrginit = Cptgininode(yrightt, clsnum)
    maxnum += 1
    Nodes[maxnum] = Node(maxnum)
    Nodes = Givetraintonode(Nodes, maxnum, Rightidx)
    Nodes = Givevaltonode(Nodes, maxnum, Rightidxt)
    yrcount = County(yright, clsnum)
    yrcountt = County(yrightt, clsnum)
    Nodes[maxnum].ycount = yrcount
    Nodes[maxnum].ycountt = yrcountt
    Nodes[maxnum].predcls = yrcountt.argmax()
    Nodes[maxnum].ginis = yrgini
    Nodes[maxnum].ginist = yrginit
    if yrginit == 0 or yrgini == 0:
        Nodes[maxnum].stoptrain = True
    else:
        Nodes[maxnum].stoptrain = False
    Nodes[pronum].rightchd = maxnum
    Nodes[maxnum].prntnb = pronum
    Nodes[maxnum].childtype = 'rightchild'
    
    return Nodes, maxnum


# Binary encoding of multi-class label
def Ecdlabel(yori, cnum):
    ynew = {}
    for c in cnum:
        yc = np.zeros(yori.shape)
        yc[yori == c] = 1
        ynew[c] = yc
    return ynew


# Gini index for classification at a node
def Cptginisplit(mds, X, y, T, clsnum):
    ginis = []
    for md in mds.values():
        Xmd_preds = md(X[:,:T], X[:,T:2*T], X[:,2*T:])
        Xmd_predsrd = torch.round(Xmd_preds)
        onesnum = torch.sum(Xmd_predsrd == 1.)
        ygroup1 = y[Xmd_predsrd == 1.]
        zerosnum = torch.sum(Xmd_predsrd == 0.)
        ygroup0 = y[Xmd_predsrd == 0.]
        ginimd = Cpt_ginigroup(onesnum, ygroup1, zerosnum, ygroup0, clsnum)
        ginis.append(ginimd)
    return ginis
   

# Gini index computation for each classifier
def Cpt_ginigroup(num1, y1, num0, y0, clsnum):
    y1prob = torch.zeros(clsnum)
    y0prob = torch.zeros(clsnum)
    y1N = len(y1)
    y0N = len(y0)
    nums = num1 + num0
    for i in range(clsnum):
        if y1N>0:
            y1prob[i] = sum(y1==i)/y1N
        if y0N>0:
            y0prob[i] = sum(y0==i)/y0N

    ginipt1 = 1 - torch.sum(y1prob**2)
    ginipt0 = 1 - torch.sum(y0prob**2)
    ginirt = (num1/nums) * ginipt1 + (num0/nums) * ginipt0
    return ginirt


# Gini index for a node
def Cptgininode(yori, clsn):
    yfrac = np.zeros(clsn)
    for i in range(clsn):
        try:
            yfrac[i] = sum(yori==i)/len(yori)
        except:
            yfrac[i] = 0
    ginin = 1 - np.sum(yfrac ** 2)
    return ginin


# Accuracy for a node phase classifier
def Cpt_Accuracy(mdl, X, y, T):
    Xpreds = mdl(X[:,:T], X[:,T:2*T], X[:,2*T:])
    Xpredsnp = Xpreds.detach().numpy()
    Xpnprd = np.round(Xpredsnp)
    trueidx = np.where(Xpnprd == 1)[0]
    falseidx = np.where(Xpnprd == 0)[0]
    accup = accuracy_score(y, Xpnprd)
    
    return Xpredsnp, accup, trueidx, falseidx


# Count the number of data in each class
def County(yori, clsnum):
    ycount = np.zeros((clsnum))
    for i in range(clsnum):
        ycount[i] = sum(yori == i)
    return ycount


# Prune a tree using validation data
def Prune_tree(Tree, Xval, yval):
    Xpredclass, tstaccu, accuuptobst, keep_list = Postprune(Tree, Xval, yval)
    Tree_pruned = {}
    
    for node_idx in Tree.keys():
        if node_idx in keep_list:
            Tree_pruned[node_idx] = Tree[node_idx]
        else:
            if Tree[node_idx].prntnb in keep_list:
                Tree_pruned[node_idx] = Tree[node_idx]
                if hasattr(Tree_pruned[node_idx], "leftchd"):
                    del Tree_pruned[node_idx].leftchd 
                if hasattr(Tree_pruned[node_idx], "rightchd"):
                    del Tree_pruned[node_idx].rightchd
                
    return Tree_pruned


# Postprune nodes of a tree classifier
def Postprune(Nodes, Xtestori, ytestori):
    Xtestori = torch.Tensor(Xtestori)
    T = int(Xtestori.shape[1]/3)
    Nodes[0].Testidx = list(range(len(ytestori))) 
    Xpredclass = np.zeros(ytestori.shape)
    testnode = 0
    Xpredupto = np.zeros(ytestori.shape)
    Xpreduptobst = Xpredupto
    accuuptobst = 0
    keep_list, prune_list = [], []
    while testnode < len(Nodes):
        if hasattr(Nodes[testnode], 'bestmodel'):
            testidx = Nodes[testnode].Testidx
            Xtest = Variable(Xtestori[testidx,:]).to(device)
            ytest = Variable(torch.Tensor((ytestori[testidx]))).to(device)
            
            Preds_testnode = Nodes[testnode].bestmodel(Xtest[:,:T],\
                            Xtest[:,T:2*T], Xtest[:, 2*T:]).cpu()
            Predsnp = Preds_testnode.detach().numpy()
            Xpred, accutest, trueidx, falseidx = Cpt_Accuracy(Nodes[testnode].bestmodel,\
                            Xtest, ytest.cpu(), T)
            Xpredrd = np.round(Xpred)
            Nodes[testnode].testtrueidx = trueidx
            Nodes[testnode].testfalseidx = falseidx
            Nodes[testnode].Xpreds = Xpredrd
            Xpredupto[Nodes[testnode].Testidx] = Xpredrd
            accuupto = accuracy_score(ytestori, Xpredupto)
            if accuupto > accuuptobst:
                accuuptobst = accuupto - 0
                keep_list.append(testnode)
            else:
                prune_list.append(testnode)
                
            Nodes[testnode].Xpredsupto = Xpredupto
            Nodes[testnode].testaccuupto = accuupto
            
            if hasattr(Nodes[testnode], 'leftchd'):
                Nodes[Nodes[testnode].leftchd].Testidx = np.array(Nodes[testnode].\
                     Testidx)[Nodes[testnode].testtrueidx]
            if hasattr(Nodes[testnode], 'rightchd'):
                Nodes[Nodes[testnode].rightchd].Testidx = np.array(Nodes[testnode].\
                     Testidx)[Nodes[testnode].testfalseidx]
        else:
            if not hasattr(Nodes[testnode], 'leftchd') and not \
                hasattr(Nodes[testnode], 'rightchd'):
               Xpredclass[Nodes[testnode].Testidx] = Nodes[testnode].predcls
        testnode += 1
    tstaccu = accuracy_score(ytestori, Xpredclass)
    if tstaccu > accuuptobst:
        keep_list = list(Nodes.keys())

    return Xpredclass, tstaccu, accuuptobst, keep_list   


# Evaluate model's performance using test data
def Evaluate_model(Nodes, Xtestori, ytestori):
    Xtestori = torch.Tensor(Xtestori)
    clsnum = max(ytestori) + 1
    T = int(Xtestori.shape[1]/3)
    Nodes[0].Testidx = list(range(len(ytestori))) 
    Xpredclass = np.zeros(ytestori.shape)
    testnode = 0
    Xpredupto = np.zeros(ytestori.shape)
    Xpreduptobst = Xpredupto
    accuuptobst = 0
    Nodes_keys = list(Nodes.keys())
    testnode_idx = 0
    while testnode_idx < len(Nodes_keys):
        testnode = Nodes_keys[testnode_idx]
        if hasattr(Nodes[testnode], 'bestmodel'):
            testidx = Nodes[testnode].Testidx
            Xtest = Xtestori[testidx,:]
            ytest = ytestori[testidx]
            Xpred, accutest, trueidx, falseidx = Cpt_Accuracy(Nodes[testnode].bestmodel,\
                            Xtest, ytest, T)
            Xpredrd = np.round(Xpred)
            Nodes[testnode].testtrueidx = trueidx
            Nodes[testnode].testfalseidx = falseidx
            Nodes[testnode].Xpreds = Xpredrd
            Xpredupto[np.array(Nodes[testnode].Testidx)[Nodes[testnode].testtrueidx]]\
                = Nodes[testnode].bstmdlclass
            ytfalse = ytestori[np.array(Nodes[testnode].Testidx)\
                               [Nodes[testnode].testfalseidx]]
            ytfalsecount = County(ytfalse.astype(int), int(clsnum))
            ytfalsemainclass = ytfalsecount.argmax()
            Xpredupto[np.array(Nodes[testnode].Testidx)[Nodes[testnode].testfalseidx]]\
                = ytfalsemainclass
            accuupto  = accuracy_score(ytestori, Xpredupto)
            Nodes[testnode].Xpredsupto = Xpredupto
            Nodes[testnode].testaccuupto = accuupto            
            
            if hasattr(Nodes[testnode], 'leftchd'):
                Nodes[Nodes[testnode].leftchd].Testidx = np.array(Nodes[testnode].\
                     Testidx)[Nodes[testnode].testtrueidx]
            if hasattr(Nodes[testnode], 'rightchd'):
                Nodes[Nodes[testnode].rightchd].Testidx = np.array(Nodes[testnode].\
                     Testidx)[Nodes[testnode].testfalseidx]
        testnode_idx += 1
    tstaccu = accuracy_score(ytestori, Xpredupto)
    
    return tstaccu

