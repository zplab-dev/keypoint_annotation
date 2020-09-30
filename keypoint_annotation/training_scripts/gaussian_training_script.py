from __future__ import absolute_import, division, print_function
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import os
import shutil
import torch
import torch.nn as nn
from collections import OrderedDict
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from torch.utils import data
from zplib.image import colorize
from zplib.curve import spline_geometry
import freeimage
import numpy
import torch

from elegant import process_images
from elegant import worm_spline
from elegant import datamodel
from elegant.torch import dataset


from keypoint_annotation import keypoint_dataloader
from keypoint_annotation import keypoint_annotation_model
from keypoint_annotation import keypoint_training
from keypoint_annotation.dataloaders import training_dataloaders
from keypoint_annotation.production import worm_datasets

##Load in Data
train = datamodel.Timepoints.from_file('/Volumes/lugia_array/Laird_Nicolette/deep_learning/keypoint_detection/new_api/production_dataloader_test/training_paths/train_path_os.txt')
val = datamodel.Timepoints.from_file('/Volumes/lugia_array/Laird_Nicolette/deep_learning/keypoint_detection/new_api/production_dataloader_test/training_paths/val_path_os.txt')
test = datamodel.Timepoints.from_file('/Volumes/lugia_array/Laird_Nicolette/deep_learning/keypoint_detection/new_api/production_dataloader_test/training_paths/test_path_os.txt')
print(len(train), len(val), len(test))

#model parameters
sets = ['train', 'val']
scale = [0,1,2,3]      # the number of output layer for U-net
batch_size = 5
total_epoch_num = 25 # total number of epoch in training
base_lr = 0.0005      # base learning rate/
downscale = 1
image_shape = (960,96)
covariate = 100
max_val = 100

# cpu or cuda
device ='cpu'
if torch.cuda.is_available(): device='cuda:0'

kp_map_generator = training_dataloaders.GaussianKpMap(covariate=covariate, max_val=max_val)
data_generator = training_dataloaders.WormKeypointDataset(kp_map_generator,downscale=downscale, scale=scale, image_size=image_size)

datasets = {'train': dataset.WormDataset(train, data_generator),
           'val': dataset.WormDataset(val, data_generator),
           'test': dataset.WormDataset(test, data_generator)}

dataloaders = {set_name: DataLoader(datasets[set_name], 
                                    batch_size=batch_size,
                                    shuffle=set_name=='train', num_workers = 4)
               for set_name in sets}
#dataloaders = keypoint_dataloader_maps.generate_dataloader_dict(positions, annotations, downscale=1, transform=transform, batch_size=5)
dataset_sizes = {set_name: len(datasets[set_name]) for set_name in sets}
print(dataset_sizes)

project_name = '960x96_cov100_max100'
#save_dir = './'+project_name
save_dir = '/mnt/lugia_array/Laird_Nicolette/deep_learning/keypoint_detection/new_api/production_dataloader_test/new_kp_maps/gaussian_kp/'+project_name
if not os.path.exists(save_dir): os.makedirs(save_dir)
log_filename = os.path.join(save_dir, 'train.log')

### Initialize model/loss function
#init model 
initModel = keypoint_annotation_model.WormRegModel(34, scale, pretrained=True)
initModel.to(device)

#loss function
loss = keypoint_training.LossofRegmentation(downscale=downscale, scale=scale, image_shape=image_shape)

optimizer = torch.optim.Adam([{'params': initModel.parameters()}], lr=base_lr)

# Decay LR by a factor of 0.5 every int(total_epoch_num/5) epochs
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(total_epoch_num/10), gamma=0.5)

### Train model
new_start = True
ep_time = 0

#add in the the hyperparameters to the log file
fn = open(log_filename, 'a')
fn.write('\nbase_lr: {}\t scale: {}\t start_epo: {}\t total_epoch_nums: {}\t\n'.format(
            base_lr, scale, ep_time, total_epoch_num))
fn.write('covariate: {}\t, max_val: {}\t, image_shape: {}\t, downscale: {}\t\n'.format(covariate, max_val, image_shape, downscale))
fn.write('dataset_sizes: {}:{}\t {}:{}\t {}:{}\t\n'.format('train', len(train), 'val', len(val), 'test', len(test)))
fn.write('work_dir: {}\n'.format(save_dir))
fn.close()

keypoint_training.training_wrapper(dataloaders, dataset_sizes, loss, 
	start_epo=ep_time, base_lr=base_lr, total_epoch_nums=total_epoch_num, work_dir=save_dir, device=device)