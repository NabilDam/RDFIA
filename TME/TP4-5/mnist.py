import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from tme5 import MNISTData
import torch.utils.data as data
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms




class MNIST_Model(torch.nn.Module):
    def __init__(self,nx,nh,ny):
        super(MNIST_Model,self).__init__()
        self.linear1 = torch.nn.Linear(nx,nh)
        self.activ1 = torch.nn.Tanh()
        self.linear2 = torch.nn.Linear(nh,ny)
        
        

    def forward(self,x):
        y = self.linear1(x)
        y = self.activ1(y)
        y = self.linear2(y)#.squeeze()
        return y 



class MNIST_Dataset(data.Dataset):

    def __init__(self,data,train=True):
        if train:
            self.data = data._Xtrain_th
            labels = data._Ytrain_th
        else:
            self.data = data._Xtest_th
            labels = data._Ytest_th

        _,self.labels =  torch.max(labels, 1) # must be indice and not one-hot encoder.
    def __getitem__(self,index):
        return self.data[index],self.labels[index]

    def __len__(self):
        return len(self.labels)


if __name__ == '__main__':

    # data MNIST
    data_mnist = MNISTData()
    
    N = data_mnist._Xtrain_th.shape[0]
    nx = data_mnist._Xtrain_th.shape[1]
    nh = 10
    ny = data_mnist._Ytrain_th.shape[1]


    # model 
    model = MNIST_Model(nx, nh, ny)


    # dataloader
    batch_size = 20
    shuffle_dataset = True    
    dataset_train = MNIST_Dataset(data_mnist)
    dataloader_train = data.DataLoader(dataset_train,shuffle=shuffle_dataset,batch_size=batch_size)
    dataset_test = MNIST_Dataset(data_mnist,train=False)
    dataloader_test = data.DataLoader(dataset_test,shuffle=shuffle_dataset,batch_size=batch_size) 

    # learning parameters
    n_epoch = 500
    learning_rate = 10e-3
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

    # Tensorboard
    writer = SummaryWriter()


    for ep in range(n_epoch):
        print("Epochs :",ep)
        for i,(x,y) in enumerate(dataloader_train):
            model.train()
            #y = y#.float()
            #x = x#.double()
            
            pred = model(x)
            #print(pred)
            loss = criterion(pred, y.long())
            _, indsYhat = torch.max(pred,1)
            # print('loss', loss ," Prédiction : ", pred, "y True : ",y)
            writer.add_scalar('Loss_MNIST/train', loss, ep)
            acc = int((y==indsYhat).sum())/len(y)
            writer.add_scalar('Accuracy_MNIST/train', acc, ep)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        for i,(x,y) in enumerate(dataloader_test):
            with torch.no_grad():
                model.eval()
                pred = model(x)
                loss = criterion(pred,y)
                _, indsYhat = torch.max(pred,1)
                writer.add_scalar('Loss_MNIST/validation', loss, ep)
                acc = int((y==indsYhat).sum())/len(y)
                writer.add_scalar('Accuracy_MNIST/validation', acc, ep)


    # attendre un appui sur une touche pour garder les figures
    input("done")
