import os
import random 
import anndata

import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from SCMLP import MLP
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from anndata.experimental.pytorch import AnnLoader
from tqdm import tqdm

def setup_seed(seed=12345):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_device_name():
    if torch.backends.mps.is_available():
       return "mps"
    elif torch.cuda.is_available():
       return "cuda:0" 
    else:
       return "cpu"

@torch.no_grad()
def test_epoch(model,dataloader, criterion, device, epoch):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    bar = tqdm(enumerate(dataloader), total = len(dataloader))
    for i, data in bar:
        x, y = data.X, data.obs['cluster']
        x = x.to(device)
        y = y.to(device)
        y_hat = model(x)
        loss = criterion(y_hat, y)
        _, y_pred = torch.max(y_hat.data,1)
        running_loss += loss.item()
        total += y.size(0)
        correct += (y_pred == y).sum().item()
    print(f"Epoch {epoch +1} testing  loss: {running_loss/(i+1):.4f} accuracy {100. * correct / total:.2f}%") 

def main():
    setup_seed()
    device_name = get_device_name()
    device_name = "cpu" 
    device = torch.device(device_name)

    ## load train_adata and test_adata
    adata = anndata.read_h5ad("processed.h5ad")

    indices = np.random.permutation(range(adata.n_obs))
    train_indices, test_indices = np.split(indices, [int(.8 * len(indices))])  

    adata_train = adata[train_indices]
    adata_test  = adata[test_indices]    

    encoder_cluster = LabelEncoder()
    encoder_cluster.fit(adata.obs['cluster'])

    encoders = {
       'obs' : {
           'cluster' : encoder_cluster.transform
       }
    }

    train_dataloader = AnnLoader(adata_train, batch_size=128, shuffle=True, convert=encoders)
    test_dataloader = AnnLoader(adata_test, batch_size=128, shuffle=True, convert=encoders) 

    ## train an MLP model on it
    MLP_DIMS = [128, 32] 
    n_clusters = len(adata.obs['cluster'].cat.categories)
    input_size = adata.n_vars
    model = MLP(MLP_DIMS, n_classes=n_clusters, input_size = input_size)
    model.train()
    model.to(device)

    criterion = nn.CrossEntropyLoss()  # Use CrossEntropyLoss for multi-class classification
    optimizer = optim.Adam(model.parameters())

    for epoch in range(10):
        running_loss = 0.0
        correct = 0
        total = 0
        model.train()
        bar = tqdm(enumerate(train_dataloader), total = len(train_dataloader))
        for i, data in bar:
            x, y = data.X, data.obs['cluster']
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
        print (f"Epoch {epoch + 1} training loss: {running_loss/(i + 1):.4f} accuracy {100.0 * correct / total:.2f}%")
        test_epoch(model,test_dataloader, criterion,device,epoch)
    
    torch.save(model.state_dict(), "model.pth")

if __name__ == '__main__':
    main()
