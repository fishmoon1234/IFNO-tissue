"""
@author: Huaiqian You
This file contains all numerical study settings in the [paper](https://arxiv.org/pdf/2204.00205.pdf)
"""


import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from utilities3 import *

import operator
from functools import reduce
from functools import partial

from timeit import default_timer
import scipy.io
import os
import sys
from itertools import chain

torch.manual_seed(0)
np.random.seed(0)


################################################################
# fourier layer
################################################################

class SpectralConv2d_fast(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d_fast, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #x_copy = x.clone()
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)
        #helper_one = torch.ones(x.shape).to(x.device)
        #helper_ft = torch.fft.rfft2(helper_one)
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        #out_helper_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat,
        #                            device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        #out_helper_ft[:, :, :self.modes1, :self.modes2] = \
        #    self.compl_mul2d(helper_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        #out_helper_ft[:, :, -self.modes1:, :self.modes2] = \
        #    self.compl_mul2d(helper_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        #helper = torch.fft.irfft2(out_helper_ft, s=(x.size(-2), x.size(-1)))
        return x #- helper*x_copy

class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width,nlayer):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the boundary condition of the solution + 2 locations (ux_D,uy_D, x, y)
        input shape: (batchsize, x=21, y=21, c=4)
        output: one component of the solution 
        output shape: (batchsize, x=21, y=21, c=1) 
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.fc0 = nn.Linear(4, self.width)
        self.nlayer = nlayer

        self.convlayer = nn.ModuleList([SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2).cuda() for i in range(1)])
        self.w = nn.ModuleList([nn.Conv2d(self.width, self.width, 1).cuda() for i in range(1)])


        self.fc1 = nn.Linear(self.width, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        batchsize = x.shape[0]
        size_x, size_y = x.shape[1], x.shape[2]

        grid = self.get_grid(batchsize, size_x//2, size_y, x.device)
        x = torch.cat((x[:,:size_y,:,:],x[:,size_y:,:,:], grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        coef = 1./self.nlayer

        for i in range(self.nlayer-1):
            #x1 = self.convlayer[i](x)
            x1 = self.convlayer[0](x)
            x2 = self.w[0](x)
            x = F.gelu(x1+x2)*coef +x


        x1 = self.convlayer[0](x)
        x2 = self.w[0](x)
        x = (x1+x2)*coef+x
        #print(x)
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, batchsize, size_x, size_y, device):
        gridx = torch.tensor(np.linspace(0, 5.5, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 5.5, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

################################################################
# configs
################################################################



TRAIN_PATH = './Data_overlap_ave/TVAL001_noave_byset.mat'
TEST_PATH = './Data_overlap_ave/TVAL001_noave_byset.mat'


study = int(sys.argv[1])
if study == 1:

    ##study 1: indistribution study
    trains = np.loadtxt("train_indexes_tissue.txt")
    tests = np.loadtxt("test_indexes_tissue.txt")

    trains = trains - 1
    tests = tests - 1

    trains = list(trains)
    tests = list(tests)
elif study == 2:
    ##study 2: train on set 1,2,4, test on set 3,5,6,7

    trains = list(chain(range(7718), range(11257, 15270)))
    tests = list(chain(range(7718, 11257), range(15270, 26523)))
elif study == 3:
    ##study 3: train on set 1,6,7, test on set 2-5

    trains = list(chain(range(3921), range(19445, 26523)))
    tests = list(range(3921, 19445))

elif study == 4:
    ##study 4: train on set 2-7, test on set 1
    trains = list(range(3921, 26523))
    tests = list(range(0, 3921))


ntrain = len(trains)
ntest = len(tests)

modes = 8  # truncated frequency in fourier transform
width = 16 # dimension of the representation vectors

batch_size = 20
batch_size2 = batch_size

epochs = 1000
learning_rate = 3e-3
scheduler_step = 100
scheduler_gamma = 0.5
wd = 1e-5

print(epochs, learning_rate, scheduler_step, scheduler_gamma)



runtime = np.zeros(2, )
t1 = default_timer()

sub = 1
S = 21
T_in = 10
T = 1
step = 20


################################################################
# load data
################################################################

reader = MatReader(TRAIN_PATH)
train_u = reader.read_field('u')[trains,::sub].view(ntrain,2*S,S,1)

reader = MatReader(TEST_PATH)
test_u = reader.read_field('u')[tests,::sub].view(ntest,2*S,S,1)


dataall_u = torch.cat([train_u,test_u],dim=0)





y_train = train_u.numpy()

y_test = test_u.numpy()


print(y_test.shape)

print(y_train.shape)

y_train = torch.from_numpy(np.reshape(y_train,(ntrain,2*S,S)))
y_test = torch.from_numpy(np.reshape(y_test,(ntest,2*S,S)))


###extract boudary conditions from the data, using 0 in the interior region
y_train_bd = torch.ones((ntrain,2*S,S))
bd_layer = 1

for i in range(ntrain):
    y_train_bd[i,:,:] = y_train[i,:,:].clone()
y_train_bd[:,bd_layer:S-bd_layer,bd_layer:S-bd_layer] = 0
y_train_bd[:,S+bd_layer:2*S-bd_layer,bd_layer:S-bd_layer] = 0
y_train_bd = y_train_bd.unsqueeze(3)


y_test_bd = torch.ones((ntest,2*S,S))
for i in range(ntest):
    y_test_bd[i,:,:] = y_test[i,:,:].clone()
y_test_bd[:, bd_layer:S-bd_layer,bd_layer:S-bd_layer] = 0
y_test_bd[:, S+bd_layer:2*S-bd_layer,bd_layer:S-bd_layer] = 0
y_test_bd = y_test_bd.unsqueeze(3)
bd_test = y_test_bd.clone()


##normalizers
x_normalizer = UnitGaussianNormalizer(y_train_bd)
x_train_bd = x_normalizer.encode(y_train_bd)
x_test_bd = x_normalizer.encode(y_test_bd)

y_normalizer = UnitGaussianNormalizer(y_train)
y_train = y_normalizer.encode(y_train)



xx_train = torch.cat([x_train_bd],dim=3)
xx_test = torch.cat([x_test_bd],dim=3)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(xx_train, y_train), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(xx_test, y_test), batch_size=batch_size, shuffle=False)

t2 = default_timer()

print('preprocessing finished, time used:', t2-t1)
device = torch.device('cuda')

base_dir = './2D_tissue_ifno_regularization_splituv_study%d' %(study)
if not os.path.exists(base_dir):
    os.makedirs(base_dir);

################################################################
# training and evaluation
################################################################

def scheduler(optimizer,lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer
def LR_schedule(learning_rate,steps,scheduler_step,scheduler_gamma):
    #print(steps//scheduler_step)
    return learning_rate*np.power(scheduler_gamma,(steps//scheduler_step))

myloss = LpLoss(size_average=False)
y_normalizer.cuda()
for nlayer in range(3):
    nb = 3*(2**nlayer)
    print("nlayer:%d" % nb)
    model_u = FNO2d(modes, modes, width,nb).cuda()
    model_v = FNO2d(modes, modes, width, nb).cuda()
    print(count_params(model_u)+count_params(model_v))

    if nb != 3:
        restart_nb = nb // 2
        modelu_filename_restart = '%s/tissue_2difno_modelu_nlayer%d_wd%f.ckpt' % (base_dir, restart_nb,wd)
        model_u.load_state_dict(torch.load(modelu_filename_restart))
        modelv_filename_restart = '%s/tissue_2difno_modelv_nlayer%d_wd%f.ckpt' % (base_dir, restart_nb, wd)
        model_v.load_state_dict(torch.load(modelv_filename_restart))

    optimizer = torch.optim.Adam(list(model_u.parameters())+list(model_v.parameters()), lr=learning_rate, weight_decay=wd)
    modelu_filename = '%s/tissue_2difno_modelu_nlayer%d_wd%f.ckpt' % (base_dir, nb,wd)
    modelv_filename = '%s/tissue_2difno_modelv_nlayer%d_wd%f.ckpt' % (base_dir, nb, wd)
    train_l2_full_min = 1000
    test_l2_step = 0
    test_l2_full = 0
    for ep in range(epochs):
        optimizer = scheduler(optimizer, LR_schedule(learning_rate, ep, scheduler_step, scheduler_gamma))
        model_u.train()
        model_v.train()
        t1 = default_timer()
        train_l2_step = 0
        train_l2_full = 0
        for xx, yy in train_loader:
            loss = 0
            xx = xx.to(device)
            yy = yy.to(device)

            im_u = model_u(xx)
            im_v = model_v(xx)
            im = torch.cat([im_u,im_v],dim=1)
            im = y_normalizer.decode(im.reshape(xx.shape[0], 2*S,S))
            yy = y_normalizer.decode(yy.reshape(xx.shape[0], 2*S,S))
            loss += myloss(im, yy)
            train_l2_step += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if train_l2_full_min > train_l2_step/ntrain:
            train_l2_full_min = train_l2_step/ntrain
            torch.save(model_u.state_dict(), modelu_filename)
            torch.save(model_v.state_dict(), modelv_filename)
            test_l2_step = 0
            test_l2_full = 0
            with torch.no_grad():
                for xx, yy in test_loader:
                    loss = 0
                    xx = xx.to(device)
                    yy = yy.to(device)

                    im_u = model_u(xx)
                    im_v = model_v(xx)
                    im = torch.cat([im_u, im_v], dim=1)
                    im = y_normalizer.decode(im.reshape(xx.shape[0], 2*S,S))
                    loss += myloss(im, yy.reshape(xx.shape[0], 2*S,S))
                #print(loss.item())
                    test_l2_step += loss.item()

        t2 = default_timer()
        print('depth:%d, epoch [%d/%d] running time: %.3f  current training error: %f  best training error: %f  best test error: %f' % (
        nb,ep, epochs, t2 - t1,  train_l2_step / ntrain, train_l2_full_min,
        test_l2_step / ntest))
# torch.save(model, path_model)


