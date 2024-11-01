import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import scipy
import scipy.linalg
import emulator
from emulator import Supact, Affine, Better_Attention, Better_Transformer, ResBlock, ResMLP, TRF, train

#load file from sbatch file and output dir

if "-p" in sys.argv:
    idx = sys.argv.index('-p')
    trainparampath = sys.argv[idx+1]
else:
	print('NO TRAINING SAMPLES')

if "-t" in sys.argv:
    idx = sys.argv.index('-t')
    traindvpath = sys.argv[idx+1]
else:
	print('NO TRAINING DV')

if "-q" in sys.argv:
    idx = sys.argv.index('-q')
    valiparampath = sys.argv[idx+1]
else:
	print('NO VALIDATION SAMPLES')

if "-v" in sys.argv:
    idx = sys.argv.index('-v')
    validvpath = sys.argv[idx+1]
else:
	print('NO VALIDATION DV')


if "-e" in sys.argv:
    idx = sys.argv.index('-e')
    extrainfopath = sys.argv[idx+1]
else:
	print('NO EXTRA INFO')


if "-o" in sys.argv:
    idx = sys.argv.index('-o')
    outpath = sys.argv[idx+1]
else:
	print('NO OUTPUT DIRECTORY')


##### SET UP CMB POWER SPECTRA RANGE #####

camb_ell_min          = 2
camb_ell_max          = 202
camb_ell_range        = camb_ell_max - camb_ell_min

##### PICK DEVICE ON WHICH THE MODEL WILL BE TRAINED ON. WE RECOMMEND USING GPU FOR TRAINING #####
device                = torch.device("cpu")

##### DEFINE THE BATCH-SIZE AND NUMBER OF EPOCH#####
batch_size = 20

n_epoch = 50 #just for demo, in reality you need around 500 to 700 or more epochs

##### DEFINE MODEL PARAMETERS #####
int_dim_rmlp = 10   # internal dimension of the ResMLP blocks
int_dim_trf = camb_ell_range+40# internal dimension of the Transformer block, we want the int-dim-trf to be close to the actual output size
                               # Here I just demo with output-size+40
n_channel=4  # number of channels we pick
##### LOAD UP MEAN AND STD FOR INPUT AND OUTPUT #####
extrainfo=np.load(extrainfopath, allow_pickle=True)
X_mean=torch.Tensor(extrainfo.item()['X_mean'][:,:camb_ell_range])#.to(device)
X_std=torch.Tensor(extrainfo.item()['X_std'][:,:camb_ell_range])#.to(device)
Y_mean=torch.Tensor(extrainfo.item()['Y_mean'][:,:camb_ell_range]).to(device)
Y_std=torch.Tensor(extrainfo.item()['Y_std'][:,:camb_ell_range]).to(device)

##### LOAD UP COV MAT #####
covinv=np.load('Demo/covinvTT_demo.npy',allow_pickle=True)[:camb_ell_range,:camb_ell_range]
covinv=torch.Tensor(covinv).to(device) #This is inverse of the Covariance Matrix


#load in data
n_train = 500 # Number of training data vectors
train_samples = np.load(trainparampath,allow_pickle=True)[:n_train]

validation_samples = np.load(valiparampath,allow_pickle=True)

train_data_vectors = np.load(traindvpath,allow_pickle=True)[:n_train,:camb_ell_range]

validation_data_vectors = np.load(validvpath,allow_pickle=True)[:,:camb_ell_range]
train_samples = torch.Tensor(train_samples)
train_data_vectors = torch.Tensor(train_data_vectors)
validation_samples = torch.Tensor(validation_samples)
validation_data_vectors = torch.Tensor(validation_data_vectors)
#specifying input and output dimension of our model
input_size = len(train_samples[0])
out_size = len(train_data_vectors[0])



#normalizing samples and to mean 0, std 1

X_train = (train_samples-X_mean)/X_std

X_validation = (validation_samples-X_mean)/X_std

X_train = X_train.to(torch.float32)
X_validation = X_validation.to(torch.float32)

X_mean = X_mean.to(device)
X_std = X_std.to(device)

#load the data to batches. Do not send those to device yet to save space


trainset    = TensorDataset(X_train, train_data_vectors)
validset    = TensorDataset(X_validation,validation_data_vectors)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=1)
validloader = DataLoader(validset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=1)

#Set up the model and optimizer
model = TRF(input_dim=input_size,output_dim=out_size,int_dim=int_dim_rmlp, int_trf=int_dim_trf,N_channels=n_channel)
model = nn.DataParallel(model)
model.to(device)
model = model.module.to(device)
optimizer = torch.optim.Adam(model.parameters(),weight_decay=0)

# Setting up the learning rate scheduler
reduce_lr = True#reducing learning rate on plateau
if reduce_lr==True:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=0.1,patience=5)

PATH = outpath+str(n_channel)
train(model, scheduler, optimizer, trainloader, validloader, n_epoch, covinv, device,X_mean, X_std, Y_mean, Y_std, PATH)