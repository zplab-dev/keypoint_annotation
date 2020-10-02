from __future__ import absolute_import, division, print_function
import os
import platform
import torch
import numpy
import time

from torch.utils.data import Dataset, DataLoader

from elegant import process_images
from elegant import worm_spline
from elegant import datamodel
from elegant.torch import dataset

from keypoint_annotation import keypoint_annotation_model
from keypoint_annotation import keypoint_training
from keypoint_annotation.dataloaders import training_dataloaders
from keypoint_annotation.production import worm_datasets

##Load in Data
os_type = platform.system()
print(os_type)

if os_type == 'Darwin':
  train = datamodel.Timepoints.from_file('/Volumes/lugia_array/Laird_Nicolette/deep_learning/keypoint_detection/new_api/production_dataloader_test/training_paths/train_path_os.txt')
  val = datamodel.Timepoints.from_file('/Volumes/lugia_array/Laird_Nicolette/deep_learning/keypoint_detection/new_api/production_dataloader_test/training_paths/val_path_os.txt')
  test = datamodel.Timepoints.from_file('/Volumes/lugia_array/Laird_Nicolette/deep_learning/keypoint_detection/new_api/production_dataloader_test/training_paths/test_path_os.txt')
  print(len(train), len(val), len(test))
elif os_type == 'Linux':
  train = datamodel.Timepoints.from_file('/mnt/lugia_array/Laird_Nicolette/deep_learning/keypoint_detection/new_api/production_dataloader_test/training_paths/train_path_linux.txt')
  val = datamodel.Timepoints.from_file('/mnt/lugia_array/Laird_Nicolette/deep_learning/keypoint_detection/new_api/production_dataloader_test/training_paths/val_path_linux.txt')
  test = datamodel.Timepoints.from_file('/mnt/lugia_array/Laird_Nicolette/deep_learning/keypoint_detection/new_api/production_dataloader_test/training_paths/test_path_linux.txt')
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
data_generator = training_dataloaders.WormKeypointDataset(kp_map_generator,downscale=downscale, scale=scale, image_size=image_shape)

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

project_name = '960x96_cov{}_max{}'.format(covariate, max_val)
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
start_epo = 0


log_filename = os.path.join(save_dir,'train.log')    
for i, keypoint in zip([2], ['vulva_kp']):
    since = time.time()
    print('------------------------ Training {} ------------------------'.format(keypoint))
    print('base_lr: {}\t scale: {}\t start_epo: {}\t total_epoch_nums: {}\t device: {}\t save_dir: {}\t'.format(
        base_lr, scale, start_epo, total_epoch_num, device, save_dir))
    fn = open(log_filename, 'a')
    fn.write('------------------------ Training {} ------------------------\n'.format(keypoint))
    fn.write('base_lr: {}\t scale: {}\t start_epo: {}\t total_epoch_nums: {}\t device: {}\t save_dir: {}\n'.format(
        base_lr, scale, start_epo, total_epoch_num, device, save_dir))
    fn.close()
    #initialize model
    initModel = keypoint_annotation_model.WormRegModel(34, scale, pretrained=True)
    initModel.to(device)
    #define loss function
    loss_1_to_2 = loss
    optimizer = torch.optim.Adam([{'params': initModel.parameters()}], lr=base_lr)
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(total_epoch_num/10), gamma=0.5)

    #train the model
    model_ft = keypoint_training.train_reg(initModel, dataloaders, dataset_sizes, loss_1_to_2, optimizer, exp_lr_scheduler, i, keypoint, 
        start_epo=0, num_epochs=total_epoch_num, work_dir=save_dir, device=device)

    print('----------------------------------------------------------------------------')
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    
    fn = open(log_filename,'a')
    fn.write('Training complete in {:.0f}m {:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60))
    fn.close()
