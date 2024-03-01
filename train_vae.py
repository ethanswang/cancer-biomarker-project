import os
import random 
import anndata
import torch
import torch.nn as nn

import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from SCVAE import SCVAE, scvae_mse_loss
from SCMLP import MLP
import torch.optim as optim

from anndata.experimental.pytorch import AnnLoader
from tqdm import tqdm
from math import ceil, sqrt
from torch.utils.data import TensorDataset, DataLoader

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
    for i, (x,y) in bar:
        x = x.to(device)
        y = y.to(device)
        y_hat = model(x)
        loss = criterion(y_hat, y)
        _, y_pred = torch.max(y_hat.data,1)
        running_loss += loss.item()
        total += y.size(0)
        correct += (y_pred == y).sum().item()
    accuracy = 100. * correct / total
    print(f"Epoch {epoch +1} testing  loss: {running_loss/(i+1):.4f} accuracy {accuracy:.2f}%") 
    return accuracy

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

    ## train a VAE + MLP classifier model 
    n_clusters = len(adata.obs['cluster'].cat.categories)
    input_size = adata.n_vars
    hidden_dims = [256,128]
    latent_dim = 50

    model = SCVAE(input_dim=input_size, hidden_dims = hidden_dims, latent_dim = latent_dim) 
    model.to(device)
    optimizer = optim.Adam(model.parameters())
    print("Start training VAE model to reduce features dimension ... ...")
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        bar = tqdm(enumerate(train_dataloader), total = len(train_dataloader))
        for i, data in bar:
            optimizer.zero_grad()
            input = data.X
            input.to(device)
            recon_input, mu, log_var = model(input)
            loss = scvae_mse_loss(recon_input, input, mu, log_var)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        print(f"Epoch #{epoch+1}, loss: {train_loss/len(train_dataloader):.2f}")

    model.eval()
    train_latent_space = []
    train_labels = []
    test_latent_space = []
    test_labels = []
    print(f"\n\nStart reducing features dimension to {latent_dim} ... ...")
    with torch.no_grad():
        print("Reducing training dataset")
        bar = tqdm(enumerate(train_dataloader), total = len(train_dataloader))
        for i, data in bar:
            mu, _ = model.encode(data.X)
            train_latent_space.append(mu)
            train_labels.append(data.obs['cluster'])
        print("Reducing test dataset")
        bar = tqdm(enumerate(test_dataloader), total = len(test_dataloader))
        for i, data in bar:
            mu, _ = model.encode(data.X)
            test_latent_space.append(mu)
            test_labels.append(data.obs['cluster'])
 
    train_latent_space = torch.cat(train_latent_space,dim=0)
    train_labels = torch.cat(train_labels,dim=0)

    test_latent_space = torch.cat(test_latent_space,dim=0)
    test_labels = torch.cat(test_labels,dim=0)

    train_dataset = TensorDataset(train_latent_space,train_labels)
    test_dataset = TensorDataset(test_latent_space,test_labels)

    train = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test  = DataLoader(test_dataset, batch_size=64, shuffle=True)

    print("\n\nStart training MLP classifier on latent space ... ...")

    MLP_DIMS = [32]
    mlp = MLP(MLP_DIMS, n_classes=n_clusters, input_size = latent_dim)
    mlp.to(device)

    criterion = nn.CrossEntropyLoss()  # Use CrossEntropyLoss for multi-class classification
    optimizer = optim.Adam(mlp.parameters())
    
    mlp_num_epochs = 50
    for epoch in range(mlp_num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        mlp.train()
        bar = tqdm(enumerate(train), total = len(train))
        for i, (x, y) in bar:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            outputs = mlp(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
        print (f"Epoch {epoch + 1} training loss: {running_loss/(i + 1):.4f} accuracy {100.0 * correct / total:.2f}%")
        test_epoch(mlp,test, criterion,device,epoch)
    
if __name__ == '__main__':
    main()
