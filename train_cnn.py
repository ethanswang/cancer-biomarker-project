import os
import random 
import anndata

import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from SCCNN import SCCNN
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from anndata.experimental.pytorch import AnnLoader
from tqdm import tqdm
from math import ceil, sqrt

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

def reshape_x(x, height, width):
    total_size = height * width
    curr_size = x.size(1)
    padding_size = total_size - curr_size

    padded_x = F.pad(x, (0, padding_size), "constant", 0)
    return padded_x.reshape(x.size(0),height,width).unsqueeze(1)

@torch.no_grad()
def test_epoch(model,dataloader, criterion, device, height, width, epoch):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    bar = tqdm(enumerate(dataloader), total = len(dataloader))
    for i, data in bar:
        x, y = data.X, data.obs['cluster']
        x = reshape_x(x,height,width)
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

    ## train a CNN model on it
    n_clusters = len(adata.obs['cluster'].cat.categories)
    input_size = adata.n_vars
    dim = ceil(sqrt(input_size))
    model = SCCNN(dim, dim, n_clusters)
    model.train()
    model.to(device)

    criterion = nn.CrossEntropyLoss()  # Use CrossEntropyLoss for multi-class classification
    optimizer = optim.Adam(model.parameters())
    
    best_test_accu = 0
    best_train_accu = 0
    best_epoch = -1
    sliding_window = 0
    window_allowance = 5 
    for epoch in range(50):
        running_loss = 0.0
        correct = 0
        total = 0
        model.train()
        bar = tqdm(enumerate(train_dataloader), total = len(train_dataloader))
        for i, data in bar:
            x, y = data.X, data.obs['cluster']
            x = reshape_x(x,dim,dim)
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

        train_accu = 100.0 * correct / total
        print (f"Epoch {epoch + 1} training loss: {running_loss/(i + 1):.4f} accuracy {100.0 * correct / total:.2f}%")
        test_accu = test_epoch(model,test_dataloader, criterion,device,dim,dim,epoch)
        if test_accu >= best_test_accu:
            best_test_accu = test_accu
            best_train_acc = train_accu
            best_epoch = epoch
            sliding_window = 0
        else:
            sliding_window += 1

        # do not over-fit on training dataset
        if sliding_window > window_allowance:
            print(f"Testing is not imporoving in the last {window_allowance} epochs. Stop training to avoid overfitting")
            break

    print(f"Best epoch {best_epoch} with training accuracy {best_train_accu:.2f}% and testing accuracy {best_test_accu:.2f}%") 
    torch.save(model.state_dict(), "model.pth")

if __name__ == '__main__':
    main()
