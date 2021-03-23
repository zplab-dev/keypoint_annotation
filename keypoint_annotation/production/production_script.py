import re
import itertools
import platform
import argparse
import torch
from elegant import datamodel

from keypoint_annotation.model_metrics import model_metrics_script
from keypoint_annotation.production import production_utils

def run_predictor(experiment, model_path_root, covariate, max_val, downscale=1, image_shape=(960,96), mask_error=False, sigmoid=False, dim1D=False, pose_name='pose'):
    device ='cpu'
    if torch.cuda.is_available(): device='cuda:0'
    timepoint_list = datamodel.Timepoints.from_experiments(experiment) 

    device ='cpu'
    if torch.cuda.is_available(): device='cuda:0'
    image_size = (int(image_shape[0]/downscale), int(image_shape[1]/downscale))
    if sigmoid:
        pred_id = 'pred keypoints {}x{}_slope{}_max{}'.format(image_size[0], image_size[1], covariate, max_val)
    elif dim1D:
        pred_id = 'pred keypoints {}x{}_cov{}_max{}_dim1D'.format(image_size[0], image_size[1], covariate, max_val)
    else:
        pred_id = 'pred keypoints {}x{}_cov{}_max{}'.format(image_size[0], image_size[1], covariate, max_val)
    
    if 'epochs' in model_path_root:
        m = re.search("\w+epochs", model_path_root)
        epoch_num = m.group()
        pred_id+='_'+epoch_num
    if mask_error:
        pred_id+='_mask'

    model_path_root = model_path_root
    model_paths={'ant_pharynx':model_path_root+"/ant_pharynx/bestValModel.paramOnly", 
                 'post_pharynx':model_path_root+'/post_pharynx/bestValModel.paramOnly', 
                 'vulva_class':model_path_root+'/Vulva_Classifier/bestValModel.paramOnly',
                 'vulva_kp':model_path_root+'/vulva_kp/bestValModel.paramOnly', 
                 'tail':model_path_root+'/tail/bestValModel.paramOnly'}

    for timepoint in timepoint_list:
            production_utils.predict_timepoint(timepoint, pred_id, model_paths, downscale, image_shape, sigmoid, pose_name)

    experiment.write_to_disk() 

if __name__ == "__main__":
    os_type = platform.system()
    parser = argparse.ArgumentParser()
    parser.add_argument('--sigmoid', default=False, action='store_true')
    parser.add_argument('--dim1D', default=False, action='store_true')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('exp_root', action='store', type=str)
    parser.add_argument('--pose_name', type=str, default='pose')
    args = parser.parse_args()
    epochs = None
    if args.epochs:
        epochs = args.epochs
        print(epochs)

    sigmoid = False
    dim1D = False
    if args.sigmoid:
        sigmoid = True

    if args.dim1D:
        dim1D = True

    #model parameters
    downscale = 1
    image_shape = (960,96)
    mask_error = False
    print(os_type)
    covariate = 200
    max_val = 3

    if os_type == 'Darwin':
        model_path_root = '/Volumes/lugia_array/Laird_Nicolette/deep_learning/keypoint_detection/new_api/production_dataloader_test/new_kp_maps/'
        #model_path_root = '/Volumes/lugia_array/Laird_Nicolette/deep_learning/keypoint_detection/new_api/production_dataloader_test/new_api_960x96_cov100'
    elif os_type == 'Linux':
       #model_path_root = '/mnt/lugia_array/Laird_Nicolette/deep_learning/keypoint_detection/new_api/production_dataloader_test/new_api_960x96_cov100'
       model_path_root = '/mnt/lugia_array/Laird_Nicolette/deep_learning/keypoint_detection/new_api/production_dataloader_test/new_kp_maps/'

    image_size = (int(image_shape[0]/downscale), int(image_shape[1]/downscale))
    if sigmoid:
        model_path_root += 'sigmoid_kp/'
        covariate = 0.5
        project_name = '{}x{}_slope{}_max{}'.format(image_size[0], image_size[1], covariate, max_val)
    else:
        model_path_root += 'gaussian_kp/'
        if dim1D:
            model_path_root+='1D_gaussian/'
            project_name = '{}x{}_cov{}_max{}'.format(image_size[0], image_size[1], covariate, max_val)
        else:
            model_path_root+='2D_gaussian/'
            project_name = '{}x{}_cov{}_max{}'.format(image_size[0], image_size[1], covariate, max_val)

    if epochs:
            project_name = str(epochs)+'_epochs/'+project_name
    if mask_error:
        project_name+='_mask'

    
    root_path= model_path_root+project_name

    experiment_root = args.exp_root
    pose_name = args.pose_name
    experiment = datamodel.Experiment(experiment_root)
    run_predictor(experiment, root_path, covariate, max_val, downscale, image_shape, mask_error, sigmoid, dim1D, pose_name)