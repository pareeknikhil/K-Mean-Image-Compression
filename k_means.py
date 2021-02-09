#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 12:11:53 2019

@author: nikhil
"""

from matplotlib import pyplot as io
import numpy as np
#import pandas as pd
from PIL import Image
from tqdm import tqdm
import random as random
import sys

#Image to array
img1 = io.imread("Penguins.jpg")
#print(img1.shape)
m,n,s = img1.shape
img = img1.reshape(m*n,s)


def _randCentroid(k):
    cent = np.zeros((k,s), dtype = int)
    for i in range(k):
        #print(img1[random.randint(0,m),random.randint(0,n),:])
        cent[i,:] = img[random.randint(0,m*n),:]
    return (cent)


def _clustCent(mu,img,k):
    cluster_centroid = np.zeros((m*n,1), dtype = int)
    #d = np.sqrt(np.linalg.norm(img[0].reshape(1,-1)-centroids,axis=1)**2) 
    d = np.empty((m*n,1))
    for i in range(k):
        d1 = np.sqrt(np.linalg.norm(img.reshape(-1,3)-centroids[i,:].reshape(-1,3),axis=1)**2).reshape(-1,1)
        d = np.concatenate((d,d1), axis =1)
    d = np.delete(d,[0], axis = 1)
    cluster_centroid = np.argmin(d, axis = 1).reshape(-1,1)
    return (cluster_centroid)


def _findindices():
    new_centroids = centroids*0
    for i in range(k):
        i1, j1 =  np.where(cluster_centroid==i)
        new_centroids[i,:] = _meanCalc(i1,i)
    return new_centroids
    
    
def _meanCalc(index,i):
    mu = np.average(img[index], axis = 0).reshape(1,3)
    return mu


k_values = list([2, 5, 10, 15, 20])

for k in k_values:
    print("Iterating the cycle for 20 loops---")    
    centroids = _randCentroid(k)
    #cluster_centroid = _clustCent(centroids,img)
    for i in tqdm(range(20)):
        cluster_centroid = _clustCent(centroids,img,k)
        centroids = _findindices()
        
    print("K means Clustering completed")     
    
    
    img.setflags(write=1)
    
    test = np.zeros((m*n,3), dtype = np.uint8)
    print("Starting to reproduce the resulImage")
    
    for i in tqdm(range(m*n)):
        test[i,:] = centroids[cluster_centroid[i,:],:]
        
    
    img2 = test.reshape(768,1024,3)
    array = np.zeros([768, 1024, 3], dtype = np.uint8)
    array = img2
    img3 = Image.fromarray(array)
    img3.save("resultImage"+str(k)+".png")
    print("Image reproduced successfully for k = "+str(k))
    
    
    import matplotlib.image as mpimg
    img4=mpimg.imread('resultImage'+str(k)+".png")
    imgplot = io.imshow(img4)
    io.show()








    
