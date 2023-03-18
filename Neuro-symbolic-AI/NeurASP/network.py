import torch
import torch.nn as nn
import numpy as np

class Net(nn.Module):
    # def __init__(self, n_features, output_dim):
    #     super(Net, self).__init__()
    #     self.input_features = n_features
    #     self.classifier =  nn.Sequential(
    #         nn.Linear(n_features, 2048),
    #         nn.ReLU(),
    #         nn.Linear(2048, 1024),
    #         nn.ReLU(),
    #         nn.Linear(1024, 84),
    #         nn.ReLU(),
    #         nn.Linear(84, output_dim),
    #         nn.Softmax(1)
    #     )

    def __init__(self, n_features, output_dim):
        super(Net, self).__init__()
        self.input_features = n_features
        hidden_sizes = [16, 64] # 2 hidden layers

        self.classifier =  nn.Sequential(
            nn.Linear(n_features, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], output_dim),
            nn.Softmax(dim=1)
        )    

    def forward(self, x):
        x = x.view(-1, self.input_features)
        x = self.classifier(x)
        return x

# the function to test a neural network model using a test data loader
def testNN(model, testLoader, device):
    """
    Return a real number "accuracy" in [0,100] which counts 1 for each data instance; 
           a real number "singleAccuracy" in [0,100] which counts 1 for each number in the label 
    @param model: a PyTorch model whose accuracy is to be checked 
    @oaram testLoader: a PyTorch dataLoader object, including (input, output) pairs for model
    """
    # set up testing mode
    model.eval()

    # check if total prediction is correct
    correct = 0
    total = 0
    # check if each single prediction is correct
    singleCorrect = 0
    singleTotal = 0

    #list to collect targets and predictions for confusion matrix
    y_target = []
    y_pred = []
    probas = []
    with torch.no_grad():
        for data, target, _ in testLoader:
            output = model(data.to(device))

            # print("The output is: ", output)
            # print("The target is: ", target)

            probas.append(output.cpu().detach().tolist())

            # print("The probassss: ", probas)
            # print("The target shape is: ", target.shape)
            # print("Output shape: ", output.shape[:-1])
            # print("Actual Output shape: ", output.shape)
            # print("Index of maximum probability: ", output.argmax(dim=-1))

            if target.shape == output.shape[:-1]:
                pred = output.argmax(dim=-1) # get the index of the max value
            elif target.shape == output.shape:
                pred = (output >= 0).int()
            else:
                print(f'Error: none considered case for output with shape {output.shape} v.s. label with shape {target.shape}')
                import sys
                sys.exit()
            target = target.to(device).view_as(pred)

            y_target = np.concatenate( (y_target, target.int().flatten().cpu() ))
            y_pred = np.concatenate( (y_pred , pred.int().flatten().cpu()) )

            correctionMatrix = (target.int() == pred.int()).view(target.shape[0], -1)
            correct += correctionMatrix.all(1).sum().item()
            total += target.shape[0]
            singleCorrect += correctionMatrix.sum().item()
            singleTotal += target.numel()
    accuracy = 100. * correct / total
    singleAccuracy = 100. * singleCorrect / singleTotal

    return accuracy, singleAccuracy, y_target, y_pred, probas