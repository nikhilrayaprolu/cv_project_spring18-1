
# coding: utf-8

# In[5]:



import numpy as np
import argparse
import os
from random import  shuffle
from tqdm import *
import time
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

##########
# TORCH
import torch
import torchvision.models as models
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
#########


trimap_kernel = [val for val in range(20,40)]
g_mean = np.array(([123.9981, 113.70789, 102.37821])).reshape([1,1,3])


import numpy as np
import random
from scipy import misc,ndimage
import copy
import itertools
import os
from sys import getrefcount
import gc

def load_alphamatting_path(test_alpha):
    rgb_path = os.path.join(test_alpha,'merged')
    trimap_path = os.path.join(test_alpha,'trimaps/')
    alpha_path = os.path.join(test_alpha,'mask')	
    images = [os.path.join(rgb_path, i) for i in sorted(os.listdir(rgb_path))]
    tri_images = [os.path.join(trimap_path, i) for i in sorted(os.listdir(trimap_path))]
    alpha_images = [os.path.join(alpha_path, i) for i in list(np.repeat(np.array(sorted(os.listdir(alpha_path))),20))]
    return images, tri_images, alpha_images, sorted(os.listdir(trimap_path)) 

def load_alphamatting_data(rgb_path,trimap_path, alpha_path ):
    rgb = misc.imread(rgb_path)
    trimap = misc.imread(trimap_path,'L')
    alpha = misc.imread(alpha_path,'L')
    alpha = np.expand_dims(misc.imresize(alpha, [320, 320]),2)/255.0
    all_shape = trimap.shape
    rgb = misc.imresize(rgb,[320,320,3])-g_mean
    trimap = misc.imresize(trimap,[320,320],interp = 'nearest').astype(np.float32)
    trimap = np.expand_dims(trimap,2)
    trimap_size = trimap.shape
    return np.array(rgb), np.array(trimap), np.array(alpha), all_shape, trimap_size




import shutil
image_size = 320

max_epochs = 1000000

#checkpoint file path
pretrained_model = False
test_dir = '/ssd_scratch/cvit/manisha/Test_set'
test_outdir = '/ssd_scratch/cvit/manisha/test_predict'
log_dir = 'matting_log'



test_RGBs, test_trimaps, test_alphas, image_paths = load_alphamatting_path(test_dir) 



def weights_init(m):
    if isinstance(m, nn.Conv2d):
        #print(m.weight.data.shape, m.bias.data.shape)
        nn.init.xavier_normal(m.weight.data)

class DeepMatting(nn.Module):
    def __init__(self):
        super(DeepMatting, self).__init__()
        batchNorm_momentum = 0.1
        self.conv1_1 = nn.Conv2d(4, 64, kernel_size=3,stride = 1, padding=1,bias=True)
        self.bn11 = nn.BatchNorm2d(64, momentum= batchNorm_momentum)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3,stride = 1, padding=1,bias=True)
        self.bn12 = nn.BatchNorm2d(64, momentum= batchNorm_momentum)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1,bias=True)
        self.bn21 = nn.BatchNorm2d(128, momentum= batchNorm_momentum)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1,bias=True)
        self.bn22 = nn.BatchNorm2d(128, momentum= batchNorm_momentum)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1,bias=True)
        self.bn31 = nn.BatchNorm2d(256, momentum= batchNorm_momentum)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1,bias=True)
        self.bn32 = nn.BatchNorm2d(256, momentum= batchNorm_momentum)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1,bias=True)
        self.bn33 = nn.BatchNorm2d(256, momentum= batchNorm_momentum)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1,bias=True)
        self.bn41 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1,bias=True)
        self.bn42 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1,bias=True)
        self.bn43 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1,bias=True)
        self.bn51 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1,bias=True)
        self.bn52 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1,bias=True)
        self.bn53 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv6_1 = nn.Conv2d(512, 4096, kernel_size=7, padding=3,bias=True)
        self.bn61 = nn.BatchNorm2d(4096, momentum= batchNorm_momentum)
        
        self.deconv6_1 = nn.Conv2d(4096, 512, kernel_size=1,bias=True)
        self.bn61d = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.deconv5_1 = nn.Conv2d(512, 512, kernel_size=5, padding=2,bias=True)
        self.bn51d = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.deconv4_1 = nn.Conv2d(512, 256, kernel_size=5, padding=2,bias=True)
        self.bn41d = nn.BatchNorm2d(256, momentum= batchNorm_momentum)
        self.deconv3_1 = nn.Conv2d(256, 128, kernel_size=5, padding=2,bias=True)
        self.bn31d = nn.BatchNorm2d(128, momentum= batchNorm_momentum)
        self.deconv2_1 = nn.Conv2d(128, 64, kernel_size=5, padding=2,bias=True)
        self.bn21d = nn.BatchNorm2d(64, momentum= batchNorm_momentum)
        self.deconv1_1 = nn.Conv2d(64, 64, kernel_size=5, padding=2,bias=True)
        self.bn11d = nn.BatchNorm2d(64, momentum= batchNorm_momentum)
        
        self.deconv1 = nn.Conv2d(64, 1, kernel_size=5, padding=2,bias=True)
        
        
    def forward(self,x, batch_trimapsT, batch_alphasT):
              # Stage 1
        x11 = F.relu(self.bn11(self.conv1_1(x)))
        x12 = F.relu(self.bn12(self.conv1_2(x11)))
        x1p, id1 = F.max_pool2d(x12,kernel_size=(2,2), stride=(2,2),return_indices=True)

        # Stage 2
        x21 = F.relu(self.bn21(self.conv2_1(x1p)))
        x22 = F.relu(self.bn22(self.conv2_2(x21)))
        x2p, id2 = F.max_pool2d(x22,kernel_size=(2,2), stride=(2,2),return_indices=True)

        # Stage 3
        x31 = F.relu(self.bn31(self.conv3_1(x2p)))
        x32 = F.relu(self.bn32(self.conv3_2(x31)))
        x33 = F.relu(self.bn33(self.conv3_3(x32)))
        x3p, id3 = F.max_pool2d(x33,kernel_size=(2,2), stride=(2,2),return_indices=True)

        # Stage 4
        x41 = F.relu(self.bn41(self.conv4_1(x3p)))
        x42 = F.relu(self.bn42(self.conv4_2(x41)))
        x43 = F.relu(self.bn43(self.conv4_3(x42)))
        x4p, id4 = F.max_pool2d(x43,kernel_size=(2,2), stride=(2,2),return_indices=True)

        # Stage 5
        x51 = F.relu(self.bn51(self.conv5_1(x4p)))
        x52 = F.relu(self.bn52(self.conv5_2(x51)))
        x53 = F.relu(self.bn53(self.conv5_3(x52)))
        x5p, id5 = F.max_pool2d(x53,kernel_size=(2,2), stride=(2,2),return_indices=True)

        # Stage 6
        x61 = F.relu(self.bn61(self.conv6_1(x5p)))

        # Stage 6d

        x61d = F.relu(self.bn61d(self.deconv6_1(x61)))


        # Stage 5d
        x5d = F.max_unpool2d(x61d,id5, kernel_size=2, stride=2)
        x51d = F.relu(self.bn51d(self.deconv5_1(x5d)))



        # Stage 4d
        x4d = F.max_unpool2d(x51d, id4, kernel_size=2, stride=2)
        x41d = F.relu(self.bn41d(self.deconv4_1(x4d)))

        # Stage 3d
        x3d = F.max_unpool2d(x41d, id3, kernel_size=2, stride=2)
        x31d = F.relu(self.bn31d(self.deconv3_1(x3d)))

        # Stage 2d
        x2d = F.max_unpool2d(x31d, id2, kernel_size=2, stride=2)
        x21d = F.relu(self.bn21d(self.deconv2_1(x2d)))

        # Stage 1d
        x1d = F.max_unpool2d(x21d, id1, kernel_size=2, stride=2)
        x12d = F.relu(self.bn11d(self.deconv1_1(x1d)))
        x11d = F.sigmoid(self.deconv1(x12d))
        pred_mattes = x11d
        alpha_diff = torch.sqrt((pred_mattes - batch_alphasT)**2 +1e-12)
        #c_diff = torch.sqrt(batch_RGBsT - raw)

        cond = torch.eq(batch_trimapsT, 128)
        #print(cond.is_cuda)
        cond = cond.type(torch.cuda.FloatTensor)
        #print(type(cond))
        #print(batch_trimapsT.shape[0])
        wl =  cond * Variable(torch.ones([batch_trimapsT.shape[0], image_size, image_size, 1]).cuda()) + ((1-cond) *  Variable(torch.zeros([batch_trimapsT.shape[0], image_size, image_size, 1]).cuda())) 
        unknown_region_size = wl.sum()
        pred_final = cond * (pred_mattes) + (1 - cond)*(batch_trimapsT/255.0)
        alpha_loss = (alpha_diff * wl).sum()/unknown_region_size
        #print(alpha_loss)
        return pred_final, alpha_loss

    def load_my_state_dict(self, model_dict):

        own_state = self.state_dict()
        #print(own_state.keys())
        own_state_keys = self.state_dict().keys()
        model_state = model_dict
        model_p = 0
        for count, name in enumerate(model_state.keys()):

            if(count % 2 == 0 and not count==0):
                model_p+=4
            #print(count, model_p)
            if count == 28:
                break
            if count == 0:
                #print(model_state[name].shape)
                own_state[own_state_keys[model_p]].copy_(torch.cat((model_state[name], torch.zeros(64,1,3,3)),1))
            else:
                if count == 26:
                    own_state[own_state_keys[model_p]].copy_(model_state[name].view((4096,512,7,7)))
                else:
                    #print(count, name)
                    #print(own_state_keys[model_p], name)
                    #print(own_state[own_state_keys[model_p]].shape, model_state[name].shape)
                    own_state[own_state_keys[model_p]].copy_(model_state[name])
            model_p+=1
                
                    
                                                           
args = {}

args['cuda'] = True
args['resume'] = 'saved_models/checkpoint.pth.tar'
args['seed'] = 1
# cuda

args['cuda'] = torch.cuda.is_available()
USE_CUDA = True
# set the seed
torch.cuda.set_device(0)
torch.manual_seed(args['seed'])
if args['cuda']:
    torch.cuda.manual_seed(args['seed'])
model = DeepMatting()
num_gpus = torch.cuda.device_count()

# if torch.cuda.device_count() > 1:
#     print("Let's use", torch.cuda.device_count(), "GPUs!")
#     model = nn.DataParallel(model)
if torch.cuda.is_available():
    model.cuda()
# define the optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-5)


def test():
    global model
    if args['resume']:
        if os.path.isfile(args['resume']):
            print("=> loading checkpoint '{}'".format(args['resume']))
            checkpoint = torch.load(args['resume'])
            args['start_epoch'] = checkpoint['epoch']
            initial_epoch = args['start_epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args['resume'], checkpoint['epoch']))

    model.eval()
    
    vali_diff = []
    # iteration over the batches
    for i in range(1000):
        #print(test_RGBs[i])
        print(i)
        RGB, trimap, test_alpha, shape_i, trimap_size = load_alphamatting_data(test_RGBs[i], test_trimaps[i], test_alphas[i])
        test_RGB = Variable(torch.Tensor(np.expand_dims(RGB,0).astype(np.float64)), volatile=True).permute(0,3,1,2).cuda()
        test_trimap = Variable(torch.Tensor(np.expand_dims(trimap,0).astype(np.float64)), volatile=True).permute(0,3,1,2).cuda()
        test_alpha = Variable(torch.Tensor(np.expand_dims(test_alpha,0).astype(np.float64)), volatile=True).permute(0,3,1,2).cuda()


        b_input = torch.cat((test_RGB, test_trimap),1)

        # predictions
        test_out, loss = model(b_input, test_trimap, test_alpha)
        pred_matte = test_out.view(320,320).data.cpu().numpy()
        vali_diff.append(loss)
        # pred_mattes = misc.imresize(test_out[0,0,:,:].data.cpu().numpy(),shape_i)
        save_name_alpha = test_outdir + '/' +test_RGBs[i].split('/')[-1]
        misc.imsave(save_name_alpha, test_out.view(320,320).data.cpu().numpy())
    print(np.mean(vali_diff))            
test()

