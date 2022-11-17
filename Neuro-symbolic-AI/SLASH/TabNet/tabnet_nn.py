from pytorch_tabnet import tab_network
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TabNetClass(tab_network.TabNet):

    def __init__(self, 
                 input_dim=140,
                 output_dim=3,
                 n_d: int = 8,
                 n_a: int = 8,
                 n_steps: int = 3,
                 gamma: float = 1.3,
                 cat_idxs = [],
                 cat_dims = [],
                 cat_emb_dim: int = 1,
                 n_independent: int = 2,
                 n_shared: int = 2,
                 epsilon: float = 1e-15,
                 virtual_batch_size=128,
                 momentum: float = 0.02,
                 mask_type: str = "sparsemax",
                 device_name: str = device,
                 lambda_sparse=1e-3
                 ):
        super(tab_network.TabNet, self).__init__()
        self.lamda_sparse = lambda_sparse
        self.tabnet =  tab_network.TabNet(input_dim=input_dim,
                    output_dim=output_dim,
                    n_d=n_d,
                    n_a=n_a,
                    n_steps=n_steps,
                    gamma=gamma,
                    cat_idxs=cat_idxs,
                    cat_dims=cat_dims,
                    cat_emb_dim=cat_emb_dim,
                    n_independent=n_independent,
                    n_shared=n_shared,
                    epsilon=epsilon,
                    virtual_batch_size=virtual_batch_size,
                    momentum=momentum,
                    mask_type=mask_type).to(device_name)

        
    def forward(self, X, sparsity_loss=False):
        """
        Make predictions for classification on a batch (valid)

        Parameters
        ----------
        X : a :tensor: `torch.Tensor`
            Input data

        Returns
        -------
        predictions : a :tensor: `torch.Tensor`
        M_loss : a :tensor: `torch.Tensor`
        """
        # print('The dimension is: ', X.shape)
        # X shape = [batch_size,(number of features)]. eg: [64, 140]
        output, M_loss = self.tabnet(X)
        predictions = torch.nn.Softmax(dim=1)(output)
        if sparsity_loss:
            return predictions, -self.lamda_sparse * M_loss
        return predictions

