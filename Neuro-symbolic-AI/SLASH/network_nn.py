import torch
import torch.nn as nn

class Net_nn(nn.Module):
    def __init__(self, n_features):
        super(Net_nn, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 6, 5),  # 6 is the output chanel size; 5 is the kernal size; 1 (channel) 28 28 -> 6 24 24
            # Convolution Output dimension = [(I - F +2 *P) / S] +1 x D > Formula1; (I)Input dimension = 28*28, (F)-Filter or kernel = 5*5, P-Padding
            # S - Strides, Depth/no.of feature maps/activation maps/output channel - (D) - 6
            # The output dimensions are = ([(28 - 5 + (2 * 0) / 1] + 1) x 6 = (24x24x6)
            nn.MaxPool2d(2, 2),  # kernal size 2; stride size 2; 6 24 24 -> 6 12 12
            nn.ReLU(True),       # inplace=True means that it will modify the input directly thus save memory
            nn.Conv2d(6, 16, 5), # 6 12 12 -> 16 8 8 => (((12-5+(2*0)/1)+1)*16)
            nn.MaxPool2d(2, 2),  # 16 8 8 -> 16 4 4
            nn.ReLU(True) 
        )
        self.classifier =  nn.Sequential(
            nn.Linear(n_features, 2048), # n_features = columns
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 84),
            nn.ReLU(),
            nn.Linear(84, 3)
            # ,
            # nn.Softmax(1)
        )

    def forward(self, x, marg_idx=None, type=1):
        
        assert type == 1, "only posterior computations are available for this network"

        # If the list of the pixel numbers to be marginalised is given,
        # then genarate a marginalisation mask from it and apply to the
        # tensor 'x'
        if marg_idx:
            batch_size = x.shape[0]
            with torch.no_grad():
                marg_mask = torch.ones_like(x, device=x.device).reshape(batch_size, 1, -1)
                marg_mask[:, :, marg_idx] = 0
                marg_mask = marg_mask.reshape_as(x)
                marg_mask.requires_grad_(False)
            x = torch.einsum('ijkl,ijkl->ijkl', x, marg_mask)
        # x = self.encoder(x)
        
        x = x.view(-1, 140) #x shape: [1, n_features] 
        x = self.classifier(x)
        return x

class Simple_nn(nn.Module):
    def __init__(self, n_features, num_class):
        super(Simple_nn, self).__init__()
        self.model =  nn.Sequential(
                nn.Linear(n_features, 2048), # n_features = columns
                nn.ReLU(),
                nn.Linear(2048, 1024),
                nn.ReLU(),
                nn.Linear(1024, 84),
                nn.ReLU(),
                nn.Linear(84, num_class) #[[-1.4308e+03,  6.8945e+03,  6.2874e+03], [-2.8680e+01,  3.6031e+01,  2.1507e+01]]
                # ,
                # nn.Softmax(1) #[[0.,  1.,  0.], [0.,  0.,  1.]] - Elements of input tensor from previous layer would be rescaled
                                #in the range of [0,1] and sum to 1.
                                # softmax(dim=1) - sum of all the elements in row = 1
                                # softmax(dim=0) - sum of all the elements in column = 1
            )

class MulticlassClassification(nn.Module):
    def __init__(self, num_feature, num_class):
        super(MulticlassClassification, self).__init__()
        
        self.layer_1 = nn.Linear(num_feature, 512)
        self.layer_2 = nn.Linear(512, 128)
        self.layer_3 = nn.Linear(128, 64)
        self.layer_out = nn.Linear(64, num_class) 
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.batchnorm1 = nn.BatchNorm1d(512)
        self.batchnorm2 = nn.BatchNorm1d(128)
        self.batchnorm3 = nn.BatchNorm1d(64)
        
    def forward(self, x):
        x = self.layer_1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        
        x = self.layer_2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.layer_3(x)
        x = self.batchnorm3(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.layer_out(x)
        
        return x