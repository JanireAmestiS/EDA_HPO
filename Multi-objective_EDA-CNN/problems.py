import numpy as np
from pymoo.core.problem import Problem
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
import torch.optim as optim

def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
              nn.BatchNorm2d(out_channels), 
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)
                
class CNN_model(nn.Module):
    def __init__(self, alpha: float, hyperparameters):
        super().__init__()
        self.num_classes = 10
        self.lista_cap_disc = []
        self.lista_cap_sequential = []
        self.hyperparameters = hyperparameters
        self.alpha = alpha
        
        for lay in range(hyperparameters['conv2D_layers']):
            if lay == 0:
                self.lista_cap_disc.append(conv_block(3, hyperparameters['amount_filters'][lay], pool= hyperparameters['pooling'][lay]))
            else:
                if hyperparameters['residual'][lay]=='no_res':
                        self.lista_cap_disc.append(conv_block(hyperparameters['amount_filters'][lay-1], hyperparameters['amount_filters'][lay], pool=hyperparameters['pooling'][lay]))

                else:
                    self.lista_cap_disc.append(nn.Sequential(conv_block(hyperparameters['amount_filters'][lay-1], hyperparameters['amount_filters'][lay], pool=False),
                                                            conv_block(hyperparameters['amount_filters'][lay], hyperparameters['amount_filters'][lay], pool=False)))
                   
        
        self.hidden_convs_disc = nn.ModuleList(self.lista_cap_disc)


        for i, l in enumerate(self.hidden_convs_disc):
            if i == 0:
                x = self.hidden_convs_disc[i](torch.rand(1, 3, 32,32))
            else:
                if self.hyperparameters['residual'][i] == 'res':
                    try:
                        x = self.hidden_convs_disc[i](x) + x
                    except:
                        x = self.hidden_convs_disc[i](x)
                else:
                    x = self.hidden_convs_disc[i](x)
        x = nn.Sequential(nn.MaxPool2d(4), nn.Flatten())(x)
        
        self.flatt = nn.Sequential(nn.MaxPool2d(4),nn.Flatten())
        for layer in range(hyperparameters['linear_layers']):
            if hyperparameters['amount_neurons'][0]==None:
                self.lista_cap_sequential.append(nn.Sequential(nn.Linear(x.shape[-1],10)))
            else:
                if layer== 0:
                    self.lista_cap_sequential.append(nn.Sequential(nn.Linear(x.shape[-1], hyperparameters['amount_neurons'][layer], bias = True), nn.ReLU(inplace=True)))
                elif layer == hyperparameters['linear_layers']-1:
                    self.lista_cap_sequential.append(nn.Linear(hyperparameters['amount_neurons'][layer-1],self.num_classes, bias = True))
                else:
                    self.lista_cap_sequential.append(nn.Linear(hyperparameters['amount_neurons'][layer-1],hyperparameters['amount_neurons'][layer], bias = True))
        self.classifier = nn.ModuleList(self.lista_cap_sequential)

    def forward(self, images: torch.Tensor, targets: torch.Tensor):
        for i, l in enumerate(self.hidden_convs_disc):
            if i == 0:
                x = self.hidden_convs_disc[i](images)
            else:
                if self.hyperparameters['residual'][i] == 'res':
                    try:
                        x = self.hidden_convs_disc[i](x) + x
                    except:
                        x = self.hidden_convs_disc[i](x)
                else:
                    x = self.hidden_convs_disc[i](x)
        x = self.flatt(x)
        for i, l in enumerate(self.classifier):
            x = self.classifier[i](x)
        return x
    
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


class Ensemble_Problem(Problem):
    def __init__(self, contador, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.psb = [[[32,32,64,64,128,128,0,0,0,0],
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
        
        self.contador = contador
        self.possible_values = [[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18],[0,1,2], [0,1,2],[0], [0], [0,1,2,3], [0,1,2,3,4], [0,1,2], [0,1,2,3,4,5,6,7,8,9,10]]
        self.variables = ['conv2D_layers', 'amount_filters','residual', 'pooling','kernel_size','strides', 'learning_rate', 'batch_size', 'optimizer', 'amount_neurons','linear_layers']

        if torch.cuda.is_available():  
            self.dev = "cuda:0" 
        else:  
            self.dev = "cpu"  
            
        self.device = torch.device(self.dev) 

        self.stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

        self.train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4, padding_mode='reflect'), 
                         transforms.RandomHorizontalFlip(), 
                         transforms.ToTensor(), 
                         transforms.Normalize(*self.stats,inplace=True)])
        self.valid_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*self.stats)])

        # training dataset and data loader
        self.train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=True, 
                                             transform=self.train_transform)
        # validation dataset and dataloader
        self.valid_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, 
                                           transform=self.valid_transform)
    def fit_CNN(self, hyperparameters, model):
        history = {'hyperparameters': [], 'train_loss':[], 'val_loss':[], 'train_acc':[], 'valid_acc':[]}
        
        train_dataloader = torch.utils.data.DataLoader(self.train_dataset, 
                                                batch_size=hyperparameters['batch_size'],
                                                shuffle=True)
        valid_dataloader = torch.utils.data.DataLoader(self.valid_dataset, 
                                                batch_size=hyperparameters['batch_size'],
                                                shuffle=False)
        criterion = nn.CrossEntropyLoss()
        
        max_lr = hyperparameters['learning_rate']
        grad_clip = 0.1
        weight_decay = 1e-4
        
        if hyperparameters['optimizer'] =='SGD':
            opt_func = optim.SGD
        elif hyperparameters['optimizer'] =='Adam':
            opt_func = optim.Adam
        elif hyperparameters['optimizer'] =='AdamW':
            opt_func = optim.AdamW

        train_div = np.ceil(50000/hyperparameters['batch_size'])
        valid_div = np.ceil(10000/hyperparameters['batch_size'])
            
        optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
        sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=20, 
                                                    steps_per_epoch=len(train_dataloader))
        
        for epoch in range(20):  

            running_loss = 0.0
            running_acc = 0.0
            valid_loss = 0.0
            valid_acc = 0.0
            
            lrs = []
            
            model.train()
            for i, data in enumerate(train_dataloader):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()

                outputs = model(inputs, labels)
                loss = criterion(outputs, labels)
                
                loss.backward()

                if grad_clip: 
                    nn.utils.clip_grad_value_(model.parameters(), grad_clip)
                
                optimizer.step()
                
                lrs.append(get_lr(optimizer))
                sched.step()
                
                

                acc = accuracy(outputs, labels)
                running_loss += loss.item()
                running_acc += acc.item()
                
            ######################    
            # validate the model #
            ######################
            model.eval()
            for k, data_val in enumerate(valid_dataloader):
                inputs_v, labels_v = data_val
                inputs_v, labels_v = inputs_v.to(self.device), labels_v.to(self.device)

                output_val = model(inputs_v, labels_v)
                loss_val = criterion(output_val, labels_v)
                val_acc = accuracy(output_val, labels_v)

                valid_loss += loss_val.item()

                valid_acc += val_acc.item()

            history['train_loss'].append(running_loss / train_div)
            history['val_loss'].append(valid_loss / valid_div)
            history['train_acc'].append(running_acc / train_div)
            history['valid_acc'].append(valid_acc / valid_div)
            history['lr'] = lrs

        print(f'train_loss: {running_loss / train_div:.3f} val_loss: {valid_loss / valid_div:.3f} train_acc: {running_acc / train_div:.3f} val_acc: {valid_acc / valid_div:.3f}')
        history['hyperparameters'].append(hyperparameters)    
        print('Finished Training')
        return history
    def _evaluate(self, x, out, *args, **kwargs):
        print('INICIO ENTRENAMIENTO N ', self.contador.individuo, 'DE LA GENERACIÓN', self.contador.generacion)    
        obj1 = []
        obj2 = []

        for solution in x:
            ts = time.time()
            solution2 = [self.psb[p][solution[p]] for p in range(len(solution))]
            solution2 = [sum(1 for num in solution2[0] if num != 0)] + solution2 + [len(solution2[-1])]

            hyperparameters = {self.variables[i]: solution2[i] for i in range(len(solution2))}
            model = CNN_model(alpha = 0.01, hyperparameters=hyperparameters)
            model.to(torch.device(self.dev))
            history = self.fit_CNN(hyperparameters, model)
            history['cost_comp'] = time.time() - ts
            history['solution'] = solution
            history['indv'] = self.contador.individuo
            history['genrc'] = self.contador.generacion
            torch.save(model, 'model_ind%s_gen%s.pth'%(self.contador.generacion, self.contador.individuo))
            self.contador.df.append(history)
            torch.save(self.contador.df, 'df_entrenamiento_MOEDA.pt')

            self.contador.aumentar_individuo()

            obj1.append(history['val_loss'][-1])
            obj2.append(history['cost_comp'])

        out['F'] = np.column_stack([np.array(obj1),np.array(obj2)])
