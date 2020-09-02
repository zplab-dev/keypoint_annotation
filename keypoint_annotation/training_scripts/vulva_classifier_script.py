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
from elegant import process_images
from elegant import worm_spline
from elegant import datamodel
from elegant.torch import dataset
import torch

from keypoint_annotation import keypoint_dataloader
from keypoint_annotation import keypoint_annotation_model
from keypoint_annotation import vulva_classifier_training
from keypoint_annotation.production import worm_datasets

### Load in Data
def has_pose(timepoint):
    pose = timepoint.annotations.get('pose', None)
    # make sure pose is not None, and center/width tcks are both not None
    return pose is not None and pose[0] is not None and pose[1] is not None
    

def has_keypoints(timepoint):
    keypoints = timepoint.annotations.get('keypoints', None)
    return keypoints is not None and not None in keypoints.values() and not False in [x in keypoints.keys() for x in ['anterior bulb', 'posterior bulb', 'vulva', 'tail']]

exp_root1 = '/mnt/lugia_array/20170919_lin-04_GFP_spe-9/'
exp_root2 = '/mnt/lugia_array/20190408_lin-4_spe-9_20C_pos-1/'
#exp_root2 = '/mnt/9karray/Mosley_Matt/20190408_lin-4_spe-9_20C_pos-1/'
exp_root3 = '/mnt/lugia_array/20190813_lin4gfp_spe9_control/20190813_lin4gfp_spe9_control/'
#exp_root3 = '/mnt/scopearray/Mosley_Matt/glp-1/20190813_lin4gfp_spe9_control'


experiments = [datamodel.Experiment(path) for path in (exp_root1, exp_root2, exp_root3)]

#filter experiments
for experiment in experiments:
    experiment.filter(timepoint_filter=(has_pose, has_keypoints))

train, val, test = datamodel.Timepoints.split_experiments(*experiments, fractions=[0.7, 0.2, 0.1])
print(len(train), len(val), len(test))

#model parameters
sets = ['train', 'val']
scale = [0,1,2,3]      # the number of output layer for U-net
batch_size = 5
total_epoch_num = 25 # total number of epoch in training
base_lr = 0.0005      # base learning rate/
downscale = 1
image_shape = (960,96)

# cpu or cuda
device ='cpu'
if torch.cuda.is_available(): device='cuda:0'

data_generator = worm_datasets.VulvaClassifier(downscale=downscale, image_size=image_shape)

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

project_name = 'Vulva_Classifier'
#save_dir = './'+project_name
save_dir = '/mnt/lugia_array/Laird_Nicolette/deep_learning/keypoint_detection/new_api/production_dataloader_test/new_api_960x96_cov100/'+project_name
print("Save Dir: ", save_dir)
if not os.path.exists(save_dir): os.makedirs(save_dir)
log_filename = os.path.join(save_dir, 'train.log')


### Initialize Model ####

#init model 
initModel = vulva_classifier_training.init_model()
initModel.to(device)

#loss function
loss_1_to_2 = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam([{'params': initModel.parameters()}], lr=base_lr)

# Decay LR by a factor of 0.5 every int(total_epoch_num/5) epochs
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(total_epoch_num/10), gamma=0.5)

### Start training ###
new_start = True
ep_time = 0

fn = open(log_filename,'w')
fn.write(log_filename+'\t'+device+'\n\n')
#fn.write(path.basename(__file__)+'\n\n')
fn.close()
file_to_note_bestModel = os.path.join(save_dir,'note_bestModel.log')
fn = open(file_to_note_bestModel, 'w')
fn.write('Record of best models on the way.\n')
fn.close()

if(not new_start):
    cur_epoch, ep_time = find_cur_epoch(save_dir)
    initModel.load_state_dict(torch.load(os.path.join(save_dir,cur_epoch)))
    best_loss = np.load('best.npy')

#add in the the hyperparameters to the log file
fn = open(log_filename, 'a')
fn.write('\nbase_lr: {}\t scale: {}\t start_epo: {}\t total_epoch_nums: {}\t\n'.format(
            base_lr, scale, ep_time, total_epoch_num))
fn.write('dataset_sizes: {}:{}\t {}:{}\t {}:{}\t'.format('train', len(train), 'val', len(val), 'test', len(test)))
fn.write('work_dir: {}\n'.format(save_dir))
fn.close()

model_ft = vulva_classifier_training.train_reg(initModel, dataloaders, dataset_sizes, loss_1_to_2, 
  optimizer,exp_lr_scheduler,start_epo=ep_time, num_epochs=total_epoch_num, work_dir=save_dir, device=device)