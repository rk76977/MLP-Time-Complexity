import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
import math
import numpy as np
from torch.utils.data import Sampler
import random
import statistics
import time
from torch.nn import init
from matplotlib.ticker import FormatStrFormatter
from PIL import Image
from torchvision import transforms
import random
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

rootdir=""
imagedir = "PASCAL-VOC-2012"

class ImageData(Dataset):
    def __init__(self, dir, I):
        self.imagedir = os.path.join(rootdir,imagedir)
        self.images =  [os.path.join(self.imagedir,file) for file in os.listdir(self.imagedir)]

        self.I = I

        self.transform = transforms.Compose([
        transforms.ToTensor(),  
        transforms.Normalize(mean=0.0,std=1.0)
        ])
        
    def __getitem__(self, idx):
        imagepath = self.images[idx]
        image = Image.open(imagepath).convert("RGB")

        normalizedTensor = self.transform(image)
        flattenedTensor = torch.flatten(normalizedTensor)
        shortenedTensor = flattenedTensor[:self.I]

        label = torch.tensor(random.randint(0, 1), dtype= torch.float).unsqueeze(0)
        
        return shortenedTensor, label
          
    def __len__(self):
        return len(self.images)
    

imageData=ImageData(rootdir,I = 50)
dataLoader=DataLoader(imageData,batch_size=1)

class Network(nn.Module):
    def __init__(self, I, N):
        super(Network, self).__init__()
        self.activation=nn.ReLU()

        self.layers = nn.ModuleList()
        numLayers = math.floor(math.log2(N)) +1


        for i in range(numLayers):
            if(i==0):
                inFeatures = I
                outFeatures = N
                # print(inFeatures,outFeatures)
                self.layers.append(nn.Linear(in_features=inFeatures,out_features=outFeatures))

            else:
                inFeatures = N
                outFeatures = math.floor(N/2.0)
                # print(inFeatures,outFeatures)
                self.layers.append(nn.Linear(in_features=inFeatures,out_features=outFeatures))

                N = math.floor(N/2.0)

    def forward(self,x):
        for layer in self.layers:
            x = layer(x)
            x=self.activation(x)
        return x
    
lossFunction=torch.nn.BCEWithLogitsLoss()

def BuildNetwork(I,N):
    imageData=ImageData(rootdir,I = I)
    dataLoader=DataLoader(imageData,batch_size=1)

    network = Network(I = I, N = N)

    return dataLoader, network

def RunNetwork(dataLoader, network, T):
    optimizer=torch.optim.SGD(network.parameters(),lr=.01,momentum=.9)

    count = 0
    
    
    forwardTotalTime = 0
    backwardTotalTime = 0
    for tensor, label in dataLoader:
        optimizer.zero_grad()

        forwardStartTime = time.time()

        logit = network(tensor)
        loss = lossFunction(logit,label)

        forwardStopTime = time.time()

        backwardStartTime = time.time()

        loss.backward()
        optimizer.step()

        backwardStopTime = time.time()
        
        forwardTotalTime += forwardStopTime - forwardStartTime
        backwardTotalTime += backwardStopTime - backwardStartTime

        count+=1
        if(count>T):
            break
    
    return forwardTotalTime, backwardTotalTime  
    

T = 10


F = np.zeros((0,1))
B = np.zeros((0,1))
I = np.zeros((0,1))
N = np.zeros((0,1))

for i in range(40,40*40+40,40):
    for n in range(100,100*40+100,100):
        dataLoader, network = BuildNetwork(I=i, N=n)
        forwardTime, backwardTime = RunNetwork(dataLoader, network, T=T)
        I = np.vstack([I,i])
        N = np.vstack([N,n])
        B = np.vstack([B,backwardTime])
        F = np.vstack([F,forwardTime])


X = np.hstack((I,N))

F_mean = np.mean(F, axis=0)
F_std = np.std(F, axis=0)
F_standardized = (F - F_mean)/F_std

B_mean = np.mean(B, axis=0)
B_std = np.std(B, axis=0)
B_standardized = (B - B_mean)/B_std

np.save(os.path.join(rootdir,"F_standardized.npy"), F_standardized)
np.save(os.path.join(rootdir,"B_standardized.npy"), B_standardized)



F = np.load(os.path.join(rootdir,"F_standardized.npy"))
B = np.load(os.path.join(rootdir,"B_standardized.npy"))
X_standardized_path = os.path.join(rootdir, 'X_standardized.npy')
X = np.load(X_standardized_path)

##Regression Forward
polynomial = PolynomialFeatures(degree=2)
X_polynomial = polynomial.fit_transform(X)

linearRegression = LinearRegression()
linearRegression.fit(X_polynomial,B)

print(linearRegression.coef_)

r2 = linearRegression.score(X_polynomial,B)


print(r2)

###Regression Forward

###Regression Backward
polynomial = PolynomialFeatures(degree=2)
X_polynomial = polynomial.fit_transform(X)

linearRegression = LinearRegression()
linearRegression.fit(X_polynomial,B)

print(linearRegression.coef_)

r2 = linearRegression.score(X_polynomial,B)

print(r2)

###Regression Backward

###Scatterplot
I_range = np.linspace(X[:,0].min(), X[:,0].max(),100)
N_range = np.linspace(X[:,1].min(), X[:,1].max(),100)


I_grid, N_grid = np.meshgrid(I_range, N_range)

grid_points = np.hstack([I_grid.reshape(-1,1), N_grid.ravel().reshape(-1,1)])

fig = plt.figure(figsize=(10,8))

axis = fig.add_subplot(111, projection = '3d')

axis.scatter(X[:,0],X[:,1], F, color = 'red', label = 'data')
axis.scatter(X[:,0],X[:,1], B, color = 'blue', label = 'data')

axis.set_xlabel("Image size (I)")
axis.set_ylabel("Bottom layer size (N)")
axis.set_zlabel("Run time")

plt.show()
###Scatterplot


###Scatterplot of differences
I_range = np.linspace(X[:,0].min(), X[:,0].max(),100)
N_range = np.linspace(X[:,1].min(), X[:,1].max(),100)

I_grid, N_grid = np.meshgrid(I_range, N_range)

grid_points = np.hstack([I_grid.reshape(-1,1), N_grid.ravel().reshape(-1,1)])

fig = plt.figure(figsize=(10,8))

axis = fig.add_subplot(111, projection = '3d')

P = F-B
positive_bool_mask = (P>=0).ravel()
P = P[positive_bool_mask]
X_positive = X[positive_bool_mask]

Q = F-B
negative_bool_mask = (Q<0).ravel()
Q = Q[negative_bool_mask]
X_negative = X[negative_bool_mask]

axis.scatter(X_positive[:,0],X_positive[:,1], P, color = 'blue', label = 'B-F ≥ 0')
axis.scatter(X_negative[:,0],X_negative[:,1], Q, color = 'red', label = 'B-F < 0')

axis.set_xlabel("Image size (I)")
axis.set_ylabel("Bottom layer size (N)")
axis.set_zlabel("Backward minus forward run time (B-F)")
plt.legend()

print(f" mean: {np.mean(P)}")

plt.show()
###Scatterplot of differences

### Plot Function
I_range = np.linspace(X[:,0].min(), X[:,0].max(),100)
N_range = np.linspace(X[:,1].min(), X[:,1].max(),100)


I_grid, N_grid = np.meshgrid(I_range, N_range)

grid_points = np.hstack([I_grid.reshape(-1,1), N_grid.ravel().reshape(-1,1)])

grid_polynomial = polynomial.transform(grid_points)

y_predictions = linearRegression.predict(grid_polynomial)


y_grid = y_predictions.reshape(I_grid.shape)

fig = plt.figure(figsize=(10,8))

axis = fig.add_subplot(111, projection = '3d')

axis.scatter(X[:,0],X[:,1], B, color = 'blue', label = 'data')


axis.plot_surface(I_grid, N_grid, y_grid, color='red', alpha = .5)

axis.set_xlabel("Input size (I)")
axis.set_ylabel("Initial layer output size (N)")
axis.set_zlabel("Backward run time (F)")

plt.show()
###
