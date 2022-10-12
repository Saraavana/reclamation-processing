# -*- coding: utf-8 -*-


from Models_node import *
from utils.datautils import *
from utils.train_utils import *



def main():    
    # Dataset_name = "Mice"
    # print('Start Training ---' + str(Dataset_name) + ' ---dataset\n')
    # dataset_path_ = "../"

    Dataset_name = "Intellizenz"
    print('Start Training ---' + str(Dataset_name) + ' ---dataset\n')
    dataset_path_ = 'C:/Saravana/Projects/Intellizenz/intellizenz-model-training/data'

    normalize_dataset = True
    Max_epoch = 100
    # model training
    Xtrain_raw, ytrain_raw, Xval_raw, yval_raw, Xtest_raw, ytest_raw \
        = Readdataset(dataset_path_, Dataset_name)
    Xtrain, Xval, Xtest = Multi_view(Xtrain_raw, Xval_raw, Xtest_raw)
    N, T = calculate_dataset_metrics(Xtrain)
    Tree = Train_model(Xtrain, Xval, ytrain_raw, yval_raw, epochs = Max_epoch,\
                       normalize_timeseries = normalize_dataset)
    # model testing
    testaccu = Evaluate_model(Tree, Xtest, ytest_raw)
    print("Test accuracy for dataset {} is --- {}".format(Dataset_name, testaccu))
    
    

if __name__ == "__main__":
    main()



