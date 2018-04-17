
import numpy as np
import argparse
import os
from random import  shuffle
import time
import random
from scipy import misc,ndimage
import copy
import itertools
from sys import getrefcount
import gc
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
f_mean = np.array(([123.9981, 113.70789, 102.37821])).reshape([1,1,3])

def UR_center(trimap):
	target = np.where(trimap==1)
	index = random.choice([i for i in range(len(target[0]))])
	return  np.array(target)[:,index][:2]

def load_path(alpha, eps, hard_mode = False):
	images_alpha = sorted(os.listdir(alpha))
	images_merged = sorted(os.listdir(eps))
	f_ix = [images_alpha.index('_'.join(i.split('_')[:-1])+'.jpg') for i in images_merged]
	images_alpha_1 = list(np.array(images_alpha)[f_ix])
	alphas_abspath = [os.path.join(alpha,common_path) for common_path in images_alpha_1]
	epses_abspath = images_merged
	epses_abspath = [os.path.join(eps,common_path) for common_path in images_merged]
	fg_path = []
	for i in range(43100):
		num = np.random.randint(0,100)
		copy1 = '_'.join(epses_abspath[i].split('_')[:-1])+'_'+str(num)+'.png'
		fg_path.append(copy1)

	return np.array(alphas_abspath),np.array(epses_abspath), np.array(fg_path)

def load_data(batch_alpha_paths, batch_eps_paths, fg):
	batch_size = batch_alpha_paths.shape[0]
	train_batch = []
	batch = 0
	while batch < batch_size:
		i = batch
		alpha = misc.imread(batch_alpha_paths[i],'L').astype(np.float32)
		eps = misc.imread(batch_eps_paths[i]).astype(np.float32)
		f_g = misc.imread(fg[i]).astype(np.float32)
		batch_i  = preprocessing_single(alpha, eps, f_g, batch_alpha_paths[i])	
		train_batch.append(batch_i)
		batch += 1
	train_batch = np.stack(train_batch).astype(np.float64)   
	return train_batch[:,:,:,:3], np.expand_dims(train_batch[:,:,:,3],3), np.expand_dims(train_batch[:,:,:,4],3), train_batch[:,:,:,8:11]

def generate_trimap(trimap,alpha):
	k_size = random.choice(trimap_kernel)
	trimap[np.where((ndimage.grey_dilation(alpha[:,:,0],size=(k_size,k_size)) - ndimage.grey_erosion(alpha[:,:,0],size=(k_size,k_size)))!=0)] = 128
	#trimap[np.where((ndimage.grey_dilation(alpha[:,:,0],size=(k_size,k_size)) - alpha[:,:,0]!=0))] = 128
	trimap = np.round(trimap/128.0)
	#np.place(trimap, trimap==0, 0)
	#np.place(trimap, trimap==128, 1)
	#np.place(trimap, trimap==255, 2)
	return trimap

def preprocessing_single(alpha, eps, f_gs, name, image_size=320):
	alpha = np.expand_dims(alpha,2)
	trimap = np.copy(alpha)
	trimap = generate_trimap(trimap,alpha)
	train_data = np.zeros([image_size,image_size,8])
	crop_size = random.choice([320,480,620])
	flip = random.choice([0,1])
	i_UR_center = UR_center(trimap)
	if (f_gs.shape != eps.shape or alpha.shape[:2] != eps.shape[:2]):
		print('warn')
		print(trimap.shape, alpha.shape, eps.shape, f_gs.shape)
	train_pre = np.concatenate([trimap, alpha, eps, f_gs],2)
	if crop_size == 320:
		h_start_index = i_UR_center[0] - 159
		if h_start_index<0:
		        h_start_index = 0
		w_start_index = i_UR_center[1] - 159
		if w_start_index<0:
		    w_start_index = 0
		tmp = train_pre[h_start_index:h_start_index+320, w_start_index:w_start_index+320, :]
		if flip:
		    tmp = tmp[:,::-1,:]

	if crop_size == 480:
		h_start_index = i_UR_center[0] - 239
		if h_start_index<0:
		    h_start_index = 0
		w_start_index = i_UR_center[1] - 239
		if w_start_index<0:
		    w_start_index = 0
		tmp = train_pre[h_start_index:h_start_index+480, w_start_index:w_start_index+480, :]
		if flip:
		    tmp = tmp[:,::-1,:]       

	if crop_size == 620:
		h_start_index = i_UR_center[0] - 309
		#boundary security
		if h_start_index<0:
		    h_start_index = 0
		w_start_index = i_UR_center[1] - 309
		if w_start_index<0:
		    w_start_index = 0
		tmp = train_pre[h_start_index:h_start_index+620, w_start_index:w_start_index+620, :]
		if flip:
		    tmp = tmp[:,::-1,:]

	tmp1 = np.zeros([image_size,image_size,8]).astype(np.float32)
	tmp1[:,:,0] = misc.imresize(tmp[:,:,0].astype(np.uint8),[image_size,image_size],interp = 'nearest',mode='L').astype(np.float32)
	tmp1[:,:,1] = misc.imresize(tmp[:,:,1].astype(np.uint8),[image_size,image_size]).astype(np.float32) / 255.0
	tmp1[:,:,2:5] = misc.imresize(tmp[:,:,2:5].astype(np.uint8),[image_size,image_size,3]).astype(np.float32)
	tmp1[:,:,5:8] = misc.imresize(tmp[:,:,5:8].astype(np.uint8),[image_size,image_size,3]).astype(np.float32) - f_mean
	raw_RGB = tmp1[:,:,2:5] 
	reduced_RGB = raw_RGB - g_mean
	tmp1 = np.concatenate([reduced_RGB, tmp1],2)
	train_data = tmp1
	train_data = train_data.astype(np.float32)
	return train_data 



class MattingDataset(Dataset):

    def __init__(self):

        """
        All required stuff happens here, loading paths, defining transformation functions and e.t.c
        """
        

    def __len__(self):
        return len(paths_alpha)

    def __getitem__(self, idx):
        batch_size = 1
        batch = idx
        batch_alpha_paths = paths_alpha[batch*batch_size:(batch+1)*batch_size]
        batch_merged_paths = paths_merged[batch*batch_size:(batch+1)*batch_size] 
        batch_fg_paths = paths_fg[batch*batch_size:(batch+1)*batch_size] 
        x1 = batch_merged_paths[0].split('_')[:-1]
        x2 = batch_fg_paths[0].split('_')[:-1]
        if (x1 != x2):
        	print(x1, x2)
        batch_RGBs, batch_trimaps, batch_alphas, batch_FGs = load_data(batch_alpha_paths, batch_merged_paths, batch_fg_paths)
        #print('inside dataloader')
        batch_RGBsT, batch_trimapsT, batch_alphasT, batch_FGsT = [(torch.Tensor(batch_RGBs.astype(np.float64))),(torch.LongTensor(batch_trimaps.astype(np.int64))),(torch.Tensor(batch_alphas.astype(np.float64))),(torch.Tensor(batch_FGs.astype(np.float64)))]

        batch_RGBsT, batch_trimapsT, batch_alphasT, batch_FGsT = [batch_RGBsT.permute(0,3,1,2), batch_trimapsT.permute(0,3,1,2), batch_alphasT.permute(0,3,1,2), batch_FGsT.permute(0,3,1,2)]
        return {'batch_RGBsT':torch.squeeze(batch_RGBsT,0), 'batch_trimapsT':torch.squeeze(batch_trimapsT,0), 'batch_alphasT':torch.squeeze(batch_alphasT,0),'batch_FGsT':torch.squeeze(batch_FGsT,0)}






def load_alphamatting_path(test_alpha):
    rgb_path = os.path.join(test_alpha,'merged')
    fg_path = os.path.join(test_alpha,'fg')
    trimap_path = os.path.join(test_alpha,'trimaps/')
    images = [os.path.join(rgb_path, i) for i in sorted(os.listdir(rgb_path))]
    fg_images = [os.path.join(fg_path, i) for i in sorted(os.listdir(fg_path))]
    fg_images = list(np.repeat(fg_images, 20))
    tri_images = [os.path.join(trimap_path, i) for i in sorted(os.listdir(trimap_path))]
    return images, tri_images, fg_images, sorted(os.listdir(trimap_path)) 

def load_alphamatting_data(rgb_path, trimap_path, fg_path):
    rgb = misc.imread(rgb_path)
    fg = misc.imread(fg_path)
    trimap = misc.imread(trimap_path,'L')
    all_shape = trimap.shape
    rgb = misc.imresize(rgb,[320,320,3])-g_mean
    fg = misc.imresize(fg,[320,320,3])-f_mean
    trimap = misc.imresize(trimap,[320,320],interp = 'nearest').astype(np.float32)
    trimap = np.round(np.expand_dims(trimap,2)/128)
    trimap_size = trimap.shape
    return np.array(rgb), np.array(trimap), np.array(fg), all_shape, trimap_size





import shutil
image_size = 320

max_epochs = 1000000

#checkpoint file path
pretrained_model = False
test_dir = 'Test_set'
test_outdir = '/test_predict'
log_dir = 'matting_log'

dataset_alpha = 'Training_set/mask'
dataset_merged = dataset_eps= 'Training_set/merged'


paths_alpha, paths_merged, paths_fg = load_path(dataset_alpha, dataset_merged, hard_mode = False)
#test_RGBs, test_trimaps, image_paths = load_alphamatting_path(test_dir) 

range_size = len(paths_alpha)
print('range_size is %d' % range_size)
#range_size/batch_size has to be int

def weights_init(m):
	if isinstance(m, nn.Conv2d):
		#print(m.weight.data.shape, m.bias.data.shape)
		nn.init.xavier_normal(m.weight.data)

class DeepMatting(nn.Module):
	def __init__(self):
		super(DeepMatting, self).__init__()
		batchNorm_momentum = 0.1
		self.conv1_1 = nn.Conv2d(6, 64, kernel_size=3,stride = 1, padding=1,bias=True)
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

		self.deconv1 = nn.Conv2d(64, 3, kernel_size=5, padding=2,bias=True)

        
	def forward(self, x, batch_trimapsT):
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
		x11d = F.softmax(self.deconv1(x12d),dim=1)	
		out = F.log_softmax(self.deconv1(x12d), dim=1)
		pred_trimaps = x11d
		#print(np.unique(batch_trimapsT.data.cpu().numpy()))
		#print(torch.squeeze(batch_trimapsT))
		#print(out)
		trimap_loss = F.nll_loss(out, torch.squeeze(batch_trimapsT,1))
		return pred_trimaps, trimap_loss


	def load_my_state_dict(self, model_dict):

		own_state = self.state_dict()
		# print(own_state.keys())
		own_state_keys = list(self.state_dict().keys())
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
				own_state[own_state_keys[model_p]].copy_(torch.cat((model_state[name], model_state[name]),1))
				#own_state[own_state_keys[model_p]].copy_(model_state[name])
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
args['resume'] = False
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

def save_checkpoint(state, is_best, filename='saved_models_FG_new/checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'saved_models_FG_new/model_best.pth.tar')

def train():
	global model
	model.train()
	model.apply(weights_init)
	initial_epoch = 0
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
	else:
	    best_prec1 = 10e6
	    print("=> no checkpoint found at '{}'".format(args['resume']))
	    #TODO_1 load_weights according to the deep matting code + match layers for loading
	    model.load_my_state_dict( models.vgg16(pretrained=True).state_dict())
	    #model.load_weights("vgg16-00b39a1b.pth") 

	matting_dataset = MattingDataset()
	dataloader = DataLoader(matting_dataset, batch_size=4,
	                    shuffle=True, num_workers=2*num_gpus)

	train_loss = []
	for epoch in range(initial_epoch, max_epochs):
		gc.collect()
		start = time.time()
		total_loss = 0
		#print(epoch, start)
		for batch, sample_batched in enumerate(dataloader):
			#print('entered loop')
			batch_RGBsT, batch_trimapsT, batch_alphasT, batch_FGsT= Variable(sample_batched['batch_RGBsT']), Variable(sample_batched['batch_trimapsT']),Variable(sample_batched['batch_alphasT']), Variable(sample_batched['batch_FGsT'])
			#print('inside epoch', epoch)
			if USE_CUDA:
				batch_RGBsT, batch_trimapsT, batch_alphasT, batch_FGsT = [batch_RGBsT.cuda(), batch_trimapsT.cuda(), batch_alphasT.cuda(), batch_FGsT.cuda()]
		   
			optimizer.zero_grad()
			b_input = torch.cat((batch_RGBsT, batch_FGsT), 1)
			#print(np.unique(batch_trimapsT.data.cpu().numpy()))
			_, trimap_loss = model(b_input, batch_trimapsT)
			trimap_loss = trimap_loss.mean()
			trimap_loss.backward()
			total_loss += trimap_loss
			optimizer.step()
			print_freq = 200
			if(batch % print_freq == 0 and not batch==0):
				print('Epoch:',epoch,'Batch:', batch, 'Loss:',total_loss/float(print_freq))
				train_loss.append(total_loss.data.cpu()/float(print_freq))
				#print(train_loss)
				plt.plot(train_loss)
				plt.savefig('test_predict/train_loss_fg_new.png')
				total_loss = 0  
				is_best = best_prec1 > total_loss/float(print_freq)
				save_checkpoint({
				     'epoch': epoch + 1,
				     'state_dict': model.state_dict(),
				     'best_prec1': total_loss/float(print_freq),
				     'optimizer' : optimizer.state_dict(),
				}, is_best)
		end = time.time()
		print('Time for 1 epoch: ', end-start)

import collections	
import scipy            
def test():
    model.eval()
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
    test_RGBs, test_trimaps, fg_paths, image_paths = load_alphamatting_path(test_dir)    
    vali_diff = []
    # iteration over the batches
    for i in range(20):
        #print('inside test')   
        RGB, trimap, fg, shape_i, trimap_size = load_alphamatting_data(test_RGBs[i], test_trimaps[i], fg_paths[i])
        print(collections.Counter(list(trimap.flatten())))
        test_RGB = Variable(torch.Tensor(np.expand_dims(RGB,0).astype(np.float64)).permute(0,3,1,2)).cuda()
        test_fg = Variable(torch.Tensor(np.expand_dims(fg,0).astype(np.float64)).permute(0,3,1,2)).cuda()
        test_trimap = Variable(torch.LongTensor(np.expand_dims(trimap,0).astype(np.int64)).permute(0,3,1,2)).cuda()
        b_input = torch.cat((test_RGB, test_fg), 1)
        #b_input = test_RGB
        # predictions
        test_out, loss = model(b_input, test_trimap)
        #pred_trimaps = misc.imresize(test_out[0,0,:,:].data.cpu().numpy(),shape_i)
        pred_trimaps = np.argmax(test_out[0,:,:,:].data.cpu().numpy(),0)
        np.place(pred_trimaps, pred_trimaps==0, 0)
        np.place(pred_trimaps, pred_trimaps==1, 128)
        np.place(pred_trimaps, pred_trimaps==2, 255)
        print(collections.Counter(list(pred_trimaps.flatten())))
        misc.imsave('test_predict/'+str(i)+'_RGB.png',RGB)
        misc.imsave('test_predict/'+str(i)+'_gt.png',trimap[:,:,0])
        # misc.imsave('test_predict/'+str(i)+'.png',pred_trimaps)
        misc.toimage(pred_trimaps, cmin=0, cmax=255).save('test_predict/'+str(i)+'.png')
        vali_diff.append(loss)
    vali_loss = np.mean(vali_diff)
    print("validation Loss:",vali_loss)

train()
#test()



