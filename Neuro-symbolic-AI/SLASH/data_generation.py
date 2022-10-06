import pandas as pd




def getDataset():
    # 1. Load the data 
    df_train = pd.read_parquet('C:/Saravana/Projects/Intellizenz/intellizenz-model-training/data/export_training_features_2016_2020.parquet.gzip')
    df_test = pd.read_parquet('C:/Saravana/Projects/Intellizenz/intellizenz-model-training/data/export_testing_features_2016_2020.parquet.gzip')
  
    #get the categorical data: 'vg_datum_year'

    # 2. Load the data and split accordingly
    train = pd.read_csv(data_file)
    target = ' <=50K'
    if "Set" not in train.columns:
        indices_file = Path(os.getcwd()+'/data/indices')
        indices_file.parent.mkdir(parents=True, exist_ok=True)
        indices = []
        if indices_file.exists():
            print("File already exists. Load the indices...")
            indices = np.load(indices_file)
        else:    
            print("Pick the indeces at random and save these as the file for the future usage.")
            indices = np.random.choice(["train", "valid", "test"], p =[.8, .1, .1], size=(train.shape[0],))
            np.save(indices_file, indices)
        train["Set"] = indices
    train_indices = train[train.Set=="train"].index
    valid_indices = train[train.Set=="valid"].index
    test_indices = train[train.Set=="test"].index
    
    # 3. Label encode categorical features and fill empty cells.
    nunique = train.nunique()
    types = train.dtypes
    categorical_columns = []
    categorical_dims =  {}
    for col in train.columns:
        if types[col] == 'object' or nunique[col] < 200:
            print(col, train[col].nunique())
            l_enc = LabelEncoder()
            train[col] = train[col].fillna("VV_likely")
            train[col] = l_enc.fit_transform(train[col].values)
            categorical_columns.append(col)
            categorical_dims[col] = len(l_enc.classes_)
        else:
            train.fillna(train.loc[train_indices, col].mean(), inplace=True)
    
    # 4. Define categorical features for categorical embeddings
    unused_feat = ['Set']
    features = [ col for col in train.columns if col not in unused_feat+[target]] 
    cat_idxs = [ i for i, f in enumerate(features) if f in categorical_columns]
    cat_dims = [ categorical_dims[f] for i, f in enumerate(features) if f in categorical_columns]
    
    # 5. Define the data subsets.
    X_train = train[features].values[train_indices]
    y_train = train[target].values[train_indices]
    X_test = train[features].values[test_indices]
    y_test = train[target].values[test_indices]
    X_valid = train[features].values[valid_indices]
    y_valid = train[target].values[valid_indices]
    
    #6. Return everything
    return cat_idxs, cat_dims, X_train, y_train, X_test, y_test, X_valid, y_valid



class Intellizenz():
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        dataTensor = self.X[index]
        query = ":- not event(p1,{}). ".format(int(self.y[index]))
        return {'t1':dataTensor}, query


# class COVTYPE(Dataset):
#     def __init__(self, root, mode):
        
#         datasets.maybe_download_covtype()

        
#         self.root = root
#         self.mode = mode
#         assert os.path.exists(root), 'Path {} does not exist'.format(root)

#         #load data from file
#         data = np.loadtxt(root, delimiter=',')        
        
#         #normalize to be in [0,1]
#         data[:,:-1] = (data[:,:-1] - data[:,:-1].min(0))/ (data[:,:-1].max(0)- data[:,:-1].min(0))
#         data[:,-1] = data[:,-1] -1 #we want our class labels from 0-6 instead of 1-7 | Forest coverage type classes
        
#         if mode == 'train':
#             self.X = torch.Tensor(data[indices[:460000],:-1])
#             self.y = torch.Tensor(data[indices[:460000],-1])
#             self.len= 460000
#         else: 
#             self.X = torch.Tensor(data[indices[460000:],:-1])
#             self.y = torch.Tensor(data[indices[460000:],-1])          
#             self.len = data.shape[0]-460000
            
#     def __getitem__(self, index):
        
#         dataTensor = self.X[index]
#         query = ":- not forest(p1,{}). ".format(int(self.y[index]))

#         return {'t1':dataTensor}, query
        
       
#     def __len__(self):
#         return self.len