import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.utils import make_grid
from tqdm import tqdm
from EDAspy.optimization import UMDAcat, EBNA
import random
import pandas as pd
import time
from torcheval.metrics import FrechetInceptionDistance
import torchvision
torch.set_printoptions(precision=10)
torch.backends.cudnn.deterministic = True
# General config
print(torch.cuda.is_available())
if torch.cuda.is_available():  
    dev = "cuda:0" 
else:  
    dev = "cpu"  
    
device = torch.device(dev) 

torch.manual_seed(0)

if torch.cuda.is_available():
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
random.seed(0)    
np.random.RandomState(0)

g_alpha     = 0.01   # LeakyReLU alpha

d_alpha = 0.01       # LeakyReLU alpha
fig = plt.figure() 
plt.show()

variables = ['discr_conv2D_layers','discr_amount_filters','discr_kernel_size','discr_strides', 'latent_dim', 'g_learning_rate', 'd_learning_rate', 'batch_size', 'optimizer']

psb = [[2], 
       [[32,16], [32,64], [64,32], [32,32],[128,128], [64,64], [16,32], [16,16], [64,128], [32,128], [128,64]], 
       [[5,5]], 
       [[2,2]], 
       [100,50,128,150,200], 
       [0.001, 0.001, 0.01, 0.00001], 
       [0.0001, 0.001, 0.01, 0.00001], 
       [64,128,256,512,1024], 
       ['Adam', 'SGD', 'AdamW']]

possible_values = [[0], [0,1,2,3,4,5,6,7,8,9, 10], [0], [0], [0,1,2,3,4], [0,1,2,3],[0,1,2,3], [0,1,2,3,4], [0,1,2]]

n_variables = len(variables)

transform = transforms.ToTensor()
dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)


class Reshape(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.reshape(-1, *self.shape)


# Generator network
class Generator(nn.Module):
    def __init__(self, alpha: float, hyperparameters, dimension):
        super().__init__()
        self.alpha = alpha
        self.lista_cap = []
        self.hyperparameters = hyperparameters
        for lay in range(hyperparameters['discr_conv2D_layers']):
            if lay != (hyperparameters['discr_conv2D_layers']-1):
                self.lista_cap.append(nn.Sequential(nn.ConvTranspose2d(in_channels = hyperparameters['discr_amount_filters'][-(lay+1)],  
                                                out_channels  = hyperparameters['discr_amount_filters'][-(lay+2)], 
                                                kernel_size=hyperparameters['discr_kernel_size'][-(lay+1)], 
                                                stride   = hyperparameters['discr_strides'][-(lay+1)], 
                                                padding= 2,
                                                output_padding=1),
                                                nn.BatchNorm2d(hyperparameters['discr_amount_filters'][-(lay+2)]),
                                                nn.LeakyReLU(alpha)))
    
        self.lista_cap.append(nn.ConvTranspose2d(in_channels = hyperparameters['discr_amount_filters'][-(lay+1)], 
                                        out_channels  = 1, 
                                        kernel_size=hyperparameters['discr_kernel_size'][-(lay+1)], 
                                        stride   = hyperparameters['discr_strides'][-(lay+1)], 
                                        padding= 2,
                                        output_padding=1))    
        self.dimension_m = dimension.numel()
   
        self.feed_forw = nn.Sequential(nn.Linear(self.hyperparameters['latent_dim'], self.dimension_m), nn.BatchNorm1d(self.dimension_m),nn.LeakyReLU(self.alpha))         
        self.reshape = Reshape(dimension[0], dimension[1], dimension[2])
        self.hidden_convs = nn.ModuleList(self.lista_cap)
        self.sigmoid = nn.Sigmoid()


    def forward(self, batch_size):
        z = torch.randn(batch_size, self.hyperparameters['latent_dim']).to(device)
        x = self.feed_forw(z)
        x = self.reshape(x)
        for i, l in enumerate(self.hidden_convs):
            x = self.hidden_convs[i](x)
        x = self.sigmoid(x)
        return x

# Discriminator network
class Discriminator(nn.Module):
    def __init__(self, alpha: float, hyperparameters):
        super().__init__()
        
        self.lista_cap_disc = []
        self.alpha = alpha

        for lay in range(hyperparameters['discr_conv2D_layers']):

            if lay ==0:
                self.lista_cap_disc.append(nn.Sequential(nn.Conv2d(in_channels = 1, 
                                                out_channels  = hyperparameters['discr_amount_filters'][lay], 
                                                kernel_size=hyperparameters['discr_kernel_size'][lay], 
                                                stride   = hyperparameters['discr_strides'][lay], 
                                                padding= 2),
                                                nn.LeakyReLU(alpha)))
            else:
                self.lista_cap_disc.append(nn.Sequential(nn.Conv2d(in_channels = hyperparameters['discr_amount_filters'][lay-1], 
                                                out_channels  = hyperparameters['discr_amount_filters'][lay], 
                                                kernel_size=hyperparameters['discr_kernel_size'][lay], 
                                                stride   = hyperparameters['discr_strides'][lay], 
                                                padding= 2),
                                                nn.BatchNorm2d(hyperparameters['discr_amount_filters'][lay]),
                                                nn.LeakyReLU(alpha)))
            
        self.hidden_convs_disc = nn.ModuleList(self.lista_cap_disc)
        
        random_tensor = torch.rand(hyperparameters['batch_size'], 1, 28, 28)
        for i, l in enumerate(self.hidden_convs_disc):
            if i == 0:
                x = self.hidden_convs_disc[i](random_tensor)
            else:
                x = self.hidden_convs_disc[i](x)
        self.dimensiones = x.shape[1:]
        self.dimens_numl = self.dimensiones.numel()
        
        
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(self.dimens_numl, self.dimens_numl),
            nn.BatchNorm1d(self.dimens_numl),
            nn.LeakyReLU(self.alpha),
            nn.Linear(self.dimens_numl, 1))
        
    def forward(self, images: torch.Tensor, targets: torch.Tensor):
        for i, l in enumerate(self.hidden_convs_disc):
            if i == 0:
                x = self.hidden_convs_disc[i](images)
            else:
                x = self.hidden_convs_disc[i](x)
        prediction = self.fc(x)
        loss = F.binary_cross_entropy_with_logits(prediction, targets)
        return loss


def save_image_grid(images: torch.Tensor, ncol: int):
    image_grid = make_grid(images, ncol)     
    image_grid = image_grid.permute(1, 2, 0) 
    image_grid = image_grid.cpu().numpy()    

    plt.imshow(image_grid, cmap = 'gray')
    plt.show()

    
def train_GAN(discriminator, generator, hyperparameters):
    dataloader = DataLoader(dataset, batch_size=hyperparameters['batch_size'], drop_last=True, worker_init_fn=torch.manual_seed(42))
    
    if hyperparameters['optimizer'] =='Adam':
        d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=hyperparameters['d_learning_rate'])
        g_optimizer = torch.optim.Adam(generator.parameters(), lr=hyperparameters['g_learning_rate'])

    elif hyperparameters['optimizer'] =='SGD':
        d_optimizer = torch.optim.SGD(discriminator.parameters(), lr=hyperparameters['d_learning_rate'], momentum= 0.9)
        g_optimizer = torch.optim.SGD(generator.parameters(), lr=hyperparameters['g_learning_rate'], momentum = 0.9)

    elif hyperparameters['optimizer'] =='AdamW':
        d_optimizer = torch.optim.AdamW(discriminator.parameters(), lr=hyperparameters['d_learning_rate'])
        g_optimizer = torch.optim.AdamW(generator.parameters(), lr=hyperparameters['g_learning_rate'])


    real_targets = torch.ones(hyperparameters['batch_size'], 1).to(device)
    fake_targets = torch.zeros(hyperparameters['batch_size'], 1).to(device)

    for epoch in range(15):

        d_losses = []
        g_losses = []
        fid_scores = []
        for i, data in enumerate(dataloader):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            #===============================
            # Discriminator training
            #===============================
            discriminator.train()
            d_loss = discriminator(images, real_targets) 
            
            generator.eval()
            with torch.no_grad():
                generated_images = generator(batch_size = hyperparameters['batch_size']) 
            loss_aa = discriminator(generated_images, fake_targets)
            d_loss += loss_aa
                
            discriminator.zero_grad()                      
            d_loss.backward()
            d_optimizer.step()

                
            #===============================
            # Generator Network Training
            #===============================

            generator.train()
            generated_images = generator(batch_size = hyperparameters['batch_size']) 
            g_loss = discriminator(generated_images, real_targets) 

            generator.zero_grad()
            g_loss.backward()
            g_optimizer.step()
            
            d_losses.append(d_loss.item())
            g_losses.append(g_loss.item())

        imagesg = generator(hyperparameters['batch_size'])
        save_image_grid(imagesg, ncol=8)

        fid = FrechetInceptionDistance()
        images = images.repeat(1, 3, 1, 1)
        images = images.to(device)
        generated_images = generated_images.repeat(1, 3, 1, 1)
        generated_images = generated_images.to(device)
        fid.update(images, is_real=True)
        fid.update(generated_images, is_real=False)
        score_fid = fid.compute()


        print('Epoch:', epoch, 'Discriminator loss:', np.mean(d_losses), 'Generator loss:', np.mean(g_losses), 'FID score:', score_fid)
    return np.mean(d_losses), np.mean(g_losses), score_fid
        

        
def define_initial_frequency(possible_values):
    """Define la frecuencia inicial para los posibles valores de cada variable."""
    frequency = []
    for values in possible_values:
        frequency.append([1/len(values)] * len(values))
    return frequency

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
def cost_function(solution: np.array, variables = variables, psb = psb, contador = contador):
    print('INICIO ENTRENAMIENTO N ', contador.individuo, 'DE LA GENERACIÓN', contador.generacion)    
    history = {}
    ts = time.time()
    solution = [psb[p][solution[p]] for p in range(len(solution))]
    hyperparameters = {variables[i]: solution[i] for i in range(len(solution))}
    print(hyperparameters)
    discriminator = Discriminator(d_alpha, hyperparameters)
    generator = Generator(g_alpha, hyperparameters, discriminator.dimensiones)    
    
    discriminator.to(torch.device("cuda:0"))
    generator.to(torch.device("cuda:0"))

    d_loss_f, g_loss_f, fid_f = train_GAN(discriminator=discriminator, generator=generator, hyperparameters=hyperparameters)

    torch.save(discriminator, 'discriminator_entrenamiento_gen%s_ind%s.pth'%(contador.generacion, contador.individuo))
    torch.save(generator, 'generator_entrenamiento_gen%s_ind%s.pth'%(contador.generacion, contador.individuo))
    tf = time.time() - ts
    
    history['d_loss'] = d_loss_f
    history['g_loss'] = g_loss_f
    history['fid'] = fid_f
    history['tiempo_ejec'] = tf
    history['solution'] = solution
    history['indv'] = contador.individuo
    history['genrc'] = contador.generacion
    contador.df.append(history)
    torch.save(contador.df, 'df_entrenamiento_EDA.pt')
        
        
    print('FIN ENTRENAMIENTO N ', contador.individuo, 'DE LA GENERACIÓN', contador.generacion, 'con d loss y g loss', d_loss_f, g_loss_f, 'FID', fid_f)    
    cost = fid_f
    contador.aumentar_individuo()
    return cost


def train_initial_CNN():
    df = pd.DataFrame(columns = ['solution', 'd_loss', 'g_loss'])
    for k in range():
        print('ENTRENAMIENTO N ', k)
        solution = [random.choice(x) for x in possible_values]
        solution2 = [psb[p][solution[p]] for p in range(len(solution))]
        hyperparameters = {variables[i]: solution2[i] for i in range(len(solution2))}
        discriminator = Discriminator(d_alpha, hyperparameters)
        generator = Generator(g_alpha, hyperparameters, discriminator.dimensiones)   
        discriminator_loss, generator_loss = train_GAN(discriminator=discriminator, generator=generator, hyperparameters=hyperparameters)
        torch.save(discriminator.state_dict(), 'discriminator_weights_entrenamiento%s.pth'%k)
        torch.save(generator.state_dict(), 'generator_weights_entrenamiento%s.pth'%k)
        df = pd.concat([df, pd.DataFrame({'solution':solution, 'd_loss': discriminator_loss, 'g_loss':generator_loss})])
        torch.save(df, 'df_entrenamiento.pt')
        print('FIN ENTRENAMIENTO N ', k, 'con d loss y g loss', discriminator_loss, generator_loss)
    return df

#Estimation of distribution Algorithm

contador = contador_generaciones()
ebna = EBNA(size_gen=50, max_iter=50, dead_iter=50, n_variables=n_variables, alpha=0.5,
                        possible_values=possible_values, frequency=define_initial_frequency(possible_values))

ebna_result = ebna.minimize(cost_function, True)

torch.save(ebna, 'ebna_model_dcgan.pt')
torch.save(ebna_result, 'ebna_result_dcgan.pt')
