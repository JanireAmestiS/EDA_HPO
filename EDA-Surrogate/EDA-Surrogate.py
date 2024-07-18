import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from torchvision import models 
from torchvision.models import resnet50, ResNet50_Weights, vgg19
from tqdm import tqdm
from EDAspy.optimization import UMDAcat, EBNA
import random
import pandas as pd
import torchvision
import time


print(torch.cuda.is_available())
if torch.cuda.is_available():  
    dev = "cuda:0" 
else:  
    dev = "cpu"  
    
device = torch.device(dev) 

print(device)
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
torch.cuda.manual_seed(0)


variables = ['conv2D_layers', 'amount_filters','residual', 'pooling','kernel_size','strides', 'learning_rate', 'batch_size', 'optimizer', 'amount_neurons','linear_layers']

#solution space
psb = [[[32,32,64,64,128,128,0,0,0,0],
        [16,16,32,0,0,0,0,0,0,0],[32,16,0,0,0,0,0,0,0,0], 
        [32,64,0,0,0,0,0,0,0,0], [64,32,0,0,0,0,0,0,0,0], [32,32,0,0,0,0,0,0,0,0],
        [64,64,0,0,0,0,0,0,0,0], [16,32,0,0,0,0,0,0,0,0],[32,128,0,0,0,0,0,0,0,0], 
        [32,32,64,64,128,128,0,0,0,0], 
        [32,32,64,64,128,128, 256,256,0,0],[32,32,64,64,128,128, 128,128,0,0],
       [32,32,64,64,128,128, 64,64,32,32], [64,64,128,128, 64,64,32,32,16,16],
       [128,128,64,64,0,0,0,0,0,0], [64,64,64,64,64,64,128,128,128,128],
       [64,128,128,256,512,512,0,0,0,0],
       [64,128,128,256,512,512,1024,512,512,128],
       [64,128,128,256,512,512,1024,2048,2048,1024]], 
       [['no_res', 'no_res','res','no_res','no_res', 'res','no_res','no_res','no_res','no_res'],
       ['no_res','no_res','no_res','no_res','no_res','no_res','no_res','no_res','no_res','no_res'],
       ['no_res','res','no_res','res','no_res','res','no_res','res','no_res','res']],
       [[False, True, False, True, True, False, False, False, False, False],
        [False,True,True,False,False, False, False, True, False,False],
       [False, False, False, False, False, False, False, False, False, False],
       [False, False, True, False, False, True, False, False, True]],
       [[3,3,3,3,3,3,3,3,3,3]], 
       [[1,1,1,1,1,1,1,1,1,1]], 
       [0.01, 0.0001, 0.001, 0.00001], 
       [16,32,64,128,256], 
       ['Adam', 'SGD', 'AdamW'],
       [[None],[32],[64], [1024],[128],[16,16], [32,32], [1024,64], [64,32], [32,16], [1024,32], [64,64]]]

possible_values = [[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18],[0,1,2], [0,1,2],[0], [0], [0,1,2,3], [0,1,2,3,4], [0,1,2], [0,1,2,3,4,5,6,7,8,9,10]]

n_variables = len(variables)-2


train_transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean=(0.5), std=(0.5))])
valid_transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean=(0.5), std=(0.5))])
stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4, padding_mode='reflect'), 
                         transforms.RandomHorizontalFlip(), 
                         transforms.ToTensor(), 
                         transforms.Normalize(*stats,inplace=True)])

valid_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*stats)])


# training dataset and data loader
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=True, 
                                             transform=train_transform)
# validation dataset and dataloader
valid_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, 
                                           transform=valid_transform)

import torch.optim as optim
import pickle 

with open('encoder_OneHot.pickle', 'rb') as f:
    encoder = pickle.load(f)
with open('scaler_MinMax.pickle', 'rb') as f:
    scaler = pickle.load(f)
with open('CV_rfr.pickle', 'rb') as f:
    CV_rfr = pickle.load(f)


class contador_generaciones():
    def __init__(self):
        super().__init__()
        self.generacion = 0
        self.individuo = 0
        self.df = []
    def aumentar_individuo(self):
        if self.individuo + 1 >=50:
            self.individuo = 0
            self.generacion += 1
        else:
            self.individuo += 1
            
contador = contador_generaciones()

def cost_function(solution: np.array, contador=contador, variables = variables, psb = psb):
    print('INICIO ENTRENAMIENTO N ', contador.individuo, 'DE LA GENERACIÓN', contador.generacion)    
    ts = time.time()
    solution2 = [psb[p][solution[p]] for p in range(len(solution))]
    solution2 = [sum(1 for num in solution2[0] if num != 0)] + solution2 + [sum(1 for num in solution2[-1] if num != 0)+1]
    hyperparameters = {variables[i]: solution2[i] for i in range(len(solution2))}
    X = pd.DataFrame([hyperparameters])
    def contar_res(lista):
        return lista.count('res')

    X['residual'] = X['residual'].apply(contar_res)
    X['pooling'] = X['pooling'].apply(sum)

    X = X.drop(['kernel_size', 'strides'], axis = 1)

    df_split = X['amount_filters'].apply(pd.Series)
    df_split.columns = [f'{X.columns[1]}_{i+1}' for i in range(df_split.shape[1])]

    X = pd.concat([X, df_split], axis=1)
    df_split = X['amount_neurons'].apply(pd.Series)
    df_split.columns = [f'{X.columns[5]}_{i+1}' for i in range(df_split.shape[1])]
    X = pd.concat([X, df_split], axis=1)
    X = X.drop(['amount_filters', 'amount_neurons'], axis = 1)
    X = X.fillna(0)
    optimizer_encoded = encoder.transform(X[['optimizer']])
    optimizer_encoded_df = pd.DataFrame(optimizer_encoded, columns=encoder.get_feature_names_out(['optimizer']))
    X = pd.concat([X, optimizer_encoded_df], axis=1)
    X.drop(['optimizer', 'optimizer_SGD'], axis=1, inplace=True)
    X = X[['conv2D_layers','learning_rate','residual','pooling','batch_size','optimizer_Adam','optimizer_AdamW', 'linear_layers']]
    X_test_esc = scaler.transform(X)
    predicciones = CV_rfr.predict(X_test_esc)

    history = {}
    history['val_loss'] = predicciones
    history['cost_comp'] = time.time() - ts
    history['solution'] = solution
    history['indv'] = contador.individuo
    history['genrc'] = contador.generacion
    contador.df.append(history)
    torch.save(contador.df, 'subrogados_EDA.pt')

    contador.aumentar_individuo()

    return history['val_loss'][-1]

def define_initial_frequency(possible_values):
    """Define la frecuencia inicial para los posibles valores de cada variable."""
    frequency = []
    for values in possible_values:
        frequency.append([1/(len(values))] * (len(values)))
    return frequency

ebna = EBNA(size_gen=50, max_iter=10, dead_iter=10, n_variables=n_variables, alpha=0.5,
                        possible_values=possible_values, frequency=define_initial_frequency(possible_values))

ebna_result = ebna.minimize(cost_function, True)