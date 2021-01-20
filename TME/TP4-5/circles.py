import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from tme5 import CirclesData

def init_params(nx, nh, ny):
    params = {}

    # TODO remplir avec les paramètres Wh, Wy, bh, by
    params["Wh"] = torch.normal(0, 0.3, size=(nx,nh),requires_grad=True)
    params["Wy"] = torch.normal(0,0.3,size=(nh,ny),requires_grad=True)
    params["bh"] = torch.normal(0,0.3,size=(nh,),requires_grad=True)
    params["by"] = torch.normal(0,0.3,size=(ny,),requires_grad=True)
    return params 


def forward(params, X):
    outputs = {}

    # TODO remplir avec les paramètres X, htilde, h, ytilde, yhat
    outputs["X"] = X
    outputs["htilde"] = torch.mm(X,params["Wh"]) + params["bh"]
    tanh = torch.nn.Tanh()
    softmax = torch.nn.Softmax(-1)
    outputs["h"] = tanh(outputs["htilde"])
    outputs["ytilde"] = torch.mm(outputs["h"],params["Wy"]) + params["by"]
    outputs["yhat"] = softmax(outputs["ytilde"])

    return outputs['yhat'], outputs

def loss_accuracy(Yhat, Y):
    L = 0
    acc = 0
    # TODO
    _, indsY = torch.max(Y, 1)
    _, indsYhat = torch.max(Yhat,1)
    acc = int((indsY==indsYhat).sum())/len(indsY)
    for i,k in enumerate(indsY) :
        L+=-torch.log(Yhat[i][k])
    L = torch.Tensor(L,requires_grad=True)
    print("L=",L)
    return L, acc

def backward(params, outputs, Y):
    grads = {}
    L,_ = loss_accuracy(outputs["yhat"],Y)
    print("Loss : ",L)
    L.backward()
    for k in params.keys():
        grads[k] = params[k].grad
    print(grads)
    # TODO remplir avec les paramètres Wy, Wh, by, bh
    # grads["Wy"] = ...

    return grads

def sgd(params, grads, eta):
    with torch.no_grad():
        for k in params.keys():
            print("nanaanana : ",k,grads[k])
            params[k]-= eta*grads[k]

    return params



if __name__ == '__main__':

    # init
    data = CirclesData()
    #data.plot_data()
    N = data.Xtrain.shape[0]
    Nbatch = 10
    nx = data.Xtrain.shape[1]
    nh = 10
    ny = data.Ytrain.shape[1]
    eta = 0.03

    # Premiers tests, code à modifier
    params = init_params(nx, nh, ny)
    Yhat, outs = forward(params, data.Xtrain)
    L, _ = loss_accuracy(Yhat, data.Ytrain)
    grads = backward(params, outs, data.Ytrain)
    params = sgd(params, grads, eta)

    nb_epoch = 1000
    batch_x =[]
    batch_y = []
    for ep in range(nb_epoch):
        for b in range(0,N-Nbatch,Nbatch):
            X = data.Xtrain[b:b+Nbatch]
            Y = data.Ytrain[b:b+Nbatch]
            Yhat, outs = forward(params, X)
            backward(params,outs,Y)
            params = sgd(params,grads,eta)


    # attendre un appui sur une touche pour garder les figures
    input("done")
