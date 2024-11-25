import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
import math
import numpy as np
import random
import time
from torch.nn import init
from matplotlib.ticker import FormatStrFormatter
from PIL import Image
from torchvision import transforms
import random
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

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
    
    Time = 0.0

    for tensor, label in dataLoader:
        optimizer.zero_grad()

        startTime = time.time()

        logit = network(tensor)
        loss = lossFunction(logit,label)
        loss.backward()
        optimizer.step()

        stopTime = time.time()
        
        Time += stopTime-startTime

        
        count+=1
        if(count>T):
            break
    
    return Time
    

T = 10


R = np.zeros((0,1))
I = np.zeros((0,1))
N = np.zeros((0,1))

for i in range(40,40*40+40,40):
    for n in range(100,100*40+100,100):
        dataLoader, network = BuildNetwork(I=i, N=n)
        runTime = RunNetwork(dataLoader, network, T=T)
        # print(i,n, runTime)
        I = np.vstack([I,i])
        N = np.vstack([N,n])
        R = np.vstack([R,runTime])


X = np.hstack((I,N))
y = R

X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X_standardized = (X - X_mean)/X_std

y_mean = np.mean(y, axis=0)
y_std = np.std(y, axis=0)
y_standardized = (y - y_mean)/y_std

np.save(os.path.join(rootdir,"X_standardized.npy"), X_standardized)
np.save(os.path.join(rootdir,"y_standardized.npy"), y_standardized)



###Regression
X_standardized_path = os.path.join(rootdir, 'X_standardized.npy')
y_standardized_path = os.path.join(rootdir, 'y_standardized.npy')

X = np.load(X_standardized_path)
y = np.load(y_standardized_path)

polynomial = PolynomialFeatures(degree=2)
X_polynomial = polynomial.fit_transform(X)


nonzero_columns = [1]
X_selected = X_polynomial[:, nonzero_columns]


linearRegression = LinearRegression()
linearRegression.fit(X_selected,y)

print(linearRegression.coef_)

r2 = linearRegression.score(X_selected,y)
print(f"R^2: {r2}")

###

feature_names = polynomial.get_feature_names_out(input_features=["I", "N"])
print(feature_names)

###Plot function


I_range = np.linspace(X[:,0].min(), X[:,0].max(),100)
N_range = np.linspace(X[:,1].min(), X[:,1].max(),100)


I_grid, N_grid = np.meshgrid(I_range, N_range)

grid_points = np.hstack([I_grid.reshape(-1,1), N_grid.ravel().reshape(-1,1)])

grid_polynomial = polynomial.transform(grid_points)

###Selected grid polynomial for 2 features
selected_grid_polynomial = grid_polynomial[:, nonzero_columns]
###

y_predictions = linearRegression.predict(selected_grid_polynomial)


y_grid = y_predictions.reshape(I_grid.shape)

fig = plt.figure(figsize=(10,8))

axis = fig.add_subplot(111, projection = '3d')

axis.scatter(X[:,0],X[:,1], y, color = 'blue', label = 'data')


axis.plot_surface(I_grid, N_grid, y_grid, color='red', alpha = .5)

axis.set_xlabel("Input size (I)")
axis.set_ylabel("Initial layer output size (N)")
axis.set_zlabel("Run time (T)")

plt.show()
exit()

###