#!/usr/bin/env python
# coding: utf-8




import matplotlib
import matplotlib.pyplot as mplt
plt = mplt
import numpy as np
from tools import *


# # Partie 1 : SIFT



# example images
I = read_grayscale('data/tools.tiff')
I2 = read_grayscale('data/Scene/CALsuburb/image_0205.jpg')
mplt.imshow(I)
#np.shape(I)


hx = np.array([1,0,1])
hy = np.array([1,2,1])
G = conv_separable(I,hx,hy)

#np.shape(G)

def compute_grad(I):
    hx = 0.25* np.array([-1,0,1])
    hy = np.array([1,2,1])
    Ix = conv_separable(I,hx,hy)
    Iy = conv_separable(I,hy,hx)
    
    return Ix, Iy

# example d'affichage du résultat

Ix, Iy = compute_grad(I)
mplt.imshow(Ix)
mplt.colorbar()
mplt.show()
mplt.imshow(Iy)
mplt.colorbar()
mplt.show()


#Ix

def compute_grad_mod_ori(I):
    Ix,Iy = compute_grad(I)
    Gn =np.sqrt(Ix**2+Iy**2)
    Go = compute_grad_ori(Ix, Iy, Gn)
    return Gn, Go



gn,go=compute_grad_mod_ori(I)
#go




def compute_sift_region(Gm, Go, mask=None):
    #print("Go :",Go)
    if mask is not None:
        g_mask = gaussian_mask() 
    else:
        g_mask = np.ones((16,16))
    Gmpond = Gm * g_mask
    P = []
    for i in range(4):
        for j in range(4):
            hist = np.zeros((8,))
            for ii in range(4):
                for jj in range(4):
                    #print("Go",Go)
                    if Go[4*j+jj][4*i+ii]!=-1:
                        hist[Go[4*j+jj][4*i+ii]]+= Gmpond[4*j+jj][4*i+ii]
            P.append(hist)
    P = np.array(P)
    P = P.reshape((128,))
    #print(P)
    #print(np.shape(P))
    # Post processing 
    
    
    norme = np.linalg.norm(P)
    
    if norme < 0.5 : 
        return np.zeros((128,))
    else : 
        P = P / norme
        for v in range(len(P)) : 
            if P[v] > 0.2 : 
                P[v] = 0.2
        norme = np.linalg.norm(P)
        P = P / norme
    #print("P: ",P)
    return P


# Example of viz of SIFTs
# set gausm to True to apply mask weighting of gradients
display_sift_region(I,           compute_grad_mod_ori, compute_sift_region, x=200, y=78, gausm=False)
display_sift_region(marche_im(), compute_grad_mod_ori, compute_sift_region, x=100, y=125, gausm=False)
display_sift_region(marche_im(), compute_grad_mod_ori, compute_sift_region, x=100, y=125, gausm=False)
display_sift_region(marche_im(), compute_grad_mod_ori, compute_sift_region, x=125, y=100, gausm=False)
display_sift_region(marche_im(), compute_grad_mod_ori, compute_sift_region, x=121, y=121, gausm=False)
display_sift_region(toy_im(),    compute_grad_mod_ori, compute_sift_region, x=95, y=95, gausm=False)


def compute_sift_image(I):
    x, y = dense_sampling(I)
    im = auto_padding(I)
    
    # TODO calculs communs aux patchs
    sifts = np.zeros((len(x), len(y), 128))
    gn, go = compute_grad_mod_ori(im)
    for i, xi in enumerate(x):
        for j, yj in enumerate(y):
            tmpgn = gn[xi:xi+16,yj:yj+16]
            tmpgo = go[xi:xi+16,yj:yj+16]
            sifts[i, j, :] = compute_sift_region(tmpgn,tmpgo) # TODO SIFT du patch de coordonnee (xi, yj)
    return sifts


compute_sift_image(I)

