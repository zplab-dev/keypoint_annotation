import os
import platform
import sys
from torch.utils import data
from zplib.image import colorize
from zplib.curve import spline_geometry
import freeimage
import numpy
from elegant import process_images
from elegant import worm_spline
from elegant import datamodel
import torch
from datetime import datetime

import matplotlib.pyplot as plt

from elegant.torch import dataset

from keypoint_annotation.production import worm_datasets
from keypoint_annotation.model_metrics import model_metrics_utils
from keypoint_annotation.production import production_utils

def run_model_metrics(model_path_root, covariate, max_val, downscale=1, image_shape=(960,96), mask_error=False, sigmoid=False):
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

    

    device ='cpu'
    if torch.cuda.is_available(): device='cuda:0'
    image_size = (int(image_shape[0]/downscale), int(image_shape[1]/downscale))
    pred_id = 'pred keypoints {}x{}_cov{}_max{}_test'.format(image_size[0], image_size[1], covariate, max_val)
    if mask_error:
        pred_id+='_mask'

    model_path_root = model_path_root
    model_paths={'ant_pharynx':model_path_root+"/ant_pharynx/bestValModel.paramOnly", 
                 'post_pharynx':model_path_root+'/post_pharynx/bestValModel.paramOnly', 
                 'vulva_class':model_path_root+'/Vulva_Classifier/bestValModel.paramOnly',
                 'vulva_kp':model_path_root+'/vulva_kp/bestValModel.paramOnly', 
                 'tail':model_path_root+'/tail/bestValModel.paramOnly'}

    #output the worst error images
    save_dir = model_path_root+"/worst_images/"
    if not os.path.exists(save_dir): os.makedirs(save_dir)

    experiments = set([tp.position.experiment for tp in test])

    exp_err = {}
	for experiment in experiments:
    	exp_err[experiment.name] = {'anterior bulb':[], 'posterior bulb':[], 'vulva':[], 'vulva class':[], 'tail':[], 'age':[]}

    

if __name__ == "__main__":
    os_type = platform.system()
    #model parameters
    downscale = 1
    image_shape = (960,96)
    mask_error = False
    print(os_type)
    try:
        covariate, max_val = sys.argv[1], sys.argv[2]
    except IndexError:
        print("Please include covariate and max_val")
        sys.exit(1)

    image_size = (int(image_shape[0]/downscale), int(image_shape[1]/downscale))
    project_name = '{}x{}_cov{}_max{}'.format(image_size[0], image_size[1], covariate, max_val)
    if mask_error:
        project_name+='_mask'
    if os_type == 'Darwin':
        model_path_root = '/Volumes/lugia_array/Laird_Nicolette/deep_learning/keypoint_detection/new_api/production_dataloader_test/new_kp_maps/gaussian_kp/'+project_name
    elif os_type == 'Linux':
        model_path_root = '/mnt/lugia_array/Laird_Nicolette/deep_learning/keypoint_detection/new_api/production_dataloader_test/new_kp_maps/gaussian_kp/'+project_name

    run_model_metrics(model_path_root, covariate, max_val, downscale, image_shape, mask_error)