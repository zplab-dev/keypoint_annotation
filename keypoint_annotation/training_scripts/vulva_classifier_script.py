from __future__ import absolute_import, division, print_function
import torchvision
import os
import platform
import torch
import torch.nn as nn
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from torch.utils import data
from zplib.image import colorize
from zplib.curve import spline_geometry

import freeimage
import numpy
import pathlib
from elegant import process_images
from elegant import worm_spline
from elegant import datamodel
from elegant.torch import dataset

from keypoint_annotation import keypoint_annotation_model
from keypoint_annotation import vulva_classifier_training
from keypoint_annotation.production import worm_datasets

def train_vulva(root_dir):
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
    image_shape = (960,64)

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
    save_dir = root_dir+'/'+project_name
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