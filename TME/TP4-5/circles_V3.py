import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from tme5 import CirclesData
import torch.utils.data as data
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms


def init_params(nx, nh, ny):
    params = {}

    # TODO remplir avec les paramètres Wh, Wy, bh, by
    params["Wh"] = torch.normal(0, 0.3, size=(nx,nh),requires_grad=True)
    params["Wy"] = torch.normal(0,0.3,size=(nh,ny),requires_grad=True)
    params["bh"] = torch.normal(0,0.3,size=(nh,),requires_grad=True)
    params["by"] = torch.normal(0,0.3,size=(ny,),requires_grad=True)
    return params 


class Circle_Model(torch.nn.Module):
    def __init__(self,nx,nh,ny):
        super(Circle_Model,self).__init__()
        self.linear1 = torch.nn.Linear(nx,nh)
        self.activ1 = torch.nn.Tanh()
        self.linear2 = torch.nn.Linear(nh,ny)
        
        

    def forward(self,x):
        y = self.linear1(x)
        y = self.activ1(y)
        y = self.linear2(y)#.squeeze()
        return y 



class Circle_Dataset(data.Dataset):

    def __init__(self,data,train=True):
        if train:
            self.data = data.Xtrain
            labels = data.Ytrain
        else:
            self.data = data.Xtest
            labels = data.Ytest

        _,self.labels =  torch.max(labels, 1) # must be indice and not one-hot encoder.
    def __getitem__(self,index):
        return self.data[index],self.labels[index]

    def __len__(self):
        return len(self.labels)


if __name__ == '__main__':

    # data circle
    data_circle = CirclesData()
    data_circle.plot_data()
    N = data_circle.Xtrain.shape[0]
    nx = data_circle.Xtrain.shape[1]
    nh = 10
    ny = data_circle.Ytrain.shape[1]


    # model 
    model = Circle_Model(nx, nh, ny)


    # dataloader
    batch_size = 50
    shuffle_dataset = True    
    dataset_train = Circle_Dataset(data_circle)
    dataloader_train = data.DataLoader(dataset_train,shuffle=shuffle_dataset,batch_size=batch_size)
    dataset_test = Circle_Dataset(data_circle,train=False)
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
            # print('loss', loss ," Prédiction : ", pred, "y True : ",y)
            _, indsYhat = torch.max(pred,1)
            acc = int((y==indsYhat).sum())/len(y)
            writer.add_scalar('Loss/train', loss, ep)
            writer.add_scalar('Accuracy/train', acc, ep)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        for i,(x,y) in enumerate(dataloader_test):
            with torch.no_grad():
                model.eval()
                pred = model(x)
                loss = criterion(pred,y)
                _, indsYhat = torch.max(pred,1)
                acc = int((y==indsYhat).sum())/len(y)
                writer.add_scalar('Loss/validation', loss, ep)
                writer.add_scalar('Accuracy/validation', acc, ep)


    # attendre un appui sur une touche pour garder les figures
    input("done")
