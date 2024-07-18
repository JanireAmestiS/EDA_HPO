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
torch.manual_seed(128)
random.seed(128)
np.random.seed(128)
torch.cuda.manual_seed(128)

class EarlyStopping:
    def __init__(self, tolerance=7, min_delta=0, min_delta2 = 0.01):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter_ovf = 0
        self.early_stop = False
        self.min_delta2 = min_delta2
        self.counter_imp = 0

    def __call__(self, train_loss, validation_loss, prev_validation_loss):
        if (validation_loss - train_loss) > self.min_delta:
            self.counter_ovf +=1
            print('overfit')
            print(self.min_delta)
            if self.counter_ovf >= self.tolerance:  
                self.early_stop = True
        if (prev_validation_loss - validation_loss) < self.min_delta2:   
            self.counter_imp += 1
            if self.counter_imp >= self.tolerance:  
                self.early_stop = True   
                
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
    
early_stopping = EarlyStopping(tolerance=5, min_delta=3, min_delta2 = 0.01)

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
def fit_CNN(hyperparameters, model):
    history = {'hyperparameters': [], 'train_loss':[], 'val_loss':[], 'train_acc':[], 'valid_acc':[]}
    
    train_dataloader = torch.utils.data.DataLoader(train_dataset, 
                                               batch_size=hyperparameters['batch_size'],
                                               shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, 
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
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=25, 
                                                steps_per_epoch=len(train_dataloader))
    
    for epoch in range(25):  

        running_loss = 0.0
        running_acc = 0.0
        valid_loss = 0.0
        valid_acc = 0.0
        
        lrs = []
        
        model.train()
        for i, data in enumerate(train_dataloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

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
            inputs_v, labels_v = inputs_v.to(device), labels_v.to(device)

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


def train_initial_CNN():
    df = []
    for k in range(20):
        print('ENTRENAMIENTO N ', k)
        solution = [random.choice(x) for x in possible_values]
        solution = [psb[p][solution[p]] for p in range(len(solution))]
        solution = [sum(1 for num in solution[0] if num != 0)] + solution + [sum(1 for num in solution[-1] if num != 0)+1]
        hyperparameters = {variables[i]: solution[i] for i in range(len(solution))}

        try:
            model = CNN_model(alpha = 0.01, hyperparameters=hyperparameters)
            model.to(torch.device(dev))
            history = fit_CNN(hyperparameters, model)
            torch.save(model, 'model_scratch%s.pth'%k)
            df.append(history)
            torch.save(df, 'df_scratch_initial_sol.pt')
        except:
            pass
        
        print('FIN ENTRENAMIENTO N ', k)
    return df

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
    solution2 = [sum(1 for num in solution2[0] if num != 0)] + solution2 + [len(solution2[-1])]

    hyperparameters = {variables[i]: solution2[i] for i in range(len(solution2))}
    print(hyperparameters)
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


    if contador.generacion != 0:
        quantl = [x['val_loss'][-1] for x in contador.df if x['genrc'] == (contador.generacion - 1)]
        quantl = np.quantile(quantl, 0.3)
        

        if predicciones[0] < quantl:
            model = CNN_model(alpha = 0.01, hyperparameters=hyperparameters)
            model.to(torch.device(dev))
            history = fit_CNN(hyperparameters, model)
            history['cost_comp'] = time.time() - ts
            history['solution'] = solution
            history['indv'] = contador.individuo
            history['genrc'] = contador.generacion
            torch.save(model, 'model_ind%s_gen%s.pth'%(contador.generacion, contador.individuo))

        else:
            history = {}
            history['val_loss'] = predicciones
            history['cost_comp'] = time.time() - ts
            history['solution'] = solution
            history['indv'] = contador.individuo
            history['genrc'] = contador.generacion
    else:
        history = {}
        history['val_loss'] = predicciones
        history['cost_comp'] = time.time() - ts
        history['solution'] = solution
        history['indv'] = contador.individuo
        history['genrc'] = contador.generacion
    contador.df.append(history)
    torch.save(contador.df, 'subrogados_CNN_EDA.pt')

    contador.aumentar_individuo()

    return history['val_loss'][-1]

def define_initial_frequency(possible_values):
    """Define la frecuencia inicial para los posibles valores de cada variable."""
    frequency = []
    for values in possible_values:
        frequency.append([1/(len(values))] * (len(values)))
    return frequency

ebna = EBNA(size_gen=50, max_iter=8, dead_iter=8, n_variables=n_variables, alpha=0.5,
                        possible_values=possible_values, frequency=define_initial_frequency(possible_values))
            
ebna_result = ebna.minimize(cost_function, True)

torch.save(ebna_result, 'ebna_result.pt')
torch.save(ebna, 'ebna_model.pt')