import os
import platform
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

from elegant.torch import dataset

from keypoint_annotation.production import worm_datasets
from keypoint_annotation.model_metrics import model_metrics_new_api
from keypoint_annotation.model_metrics import model_metrics_utils
from keypoint_annotation.production import production_utils

#model parameters
downscale = 1
image_shape = (960,96)
mask_error = False

def run_model_metrics(model_path_root, covariate, max_val):
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
    pred_id = 'pred keypoints {}x{}_cov{}_max{}_test'.format(image_shape[0], image_shape[1], covariate, max_val)
    if mask_error:
        pred_id+='_mask'

    model_path_root = model_path_root
    model_paths={'ant_pharynx':model_path_root+"/ant_pharynx/bestValModel.paramOnly", 
                 'post_pharynx':model_path_root+'/post_pharynx/bestValModel.paramOnly', 
                 'vulva_class':model_path_root+'/Vulva_Classifier/bestValModel.paramOnly',
                 'vulva_kp':model_path_root+'/vulva_kp/bestValModel.paramOnly', 
                 'tail':model_path_root+'/tail/bestValModel.paramOnly'}

    log_filename = os.path.join(model_path_root,'model_metrics.log')
    fn = open(log_filename,'a')
    time = datetime.now()
    fn.write('---------------- Model metrics run on {} ---------------------\n'.format(time))
    fn.write('Model Paths: \n')
    for k, p in model_paths.items():
        fn.write('\t{}: {}\n'.format(k,p))
    fn.close()

    #model_metrics_utils.predict_timepoint_list(test, model_paths=model_paths, pred_id=pred_id, downscale=downscale, image_shape=image_shape)

    for timepoint in test:
            model_metrics_utils.predict_timepoint(timepoint, pred_id, model_paths, downscale, image_shape)
            model_metrics_utils.predict_worst_timepoint(timepoint, 'worst case keypoints', model_paths, downscale, image_shape)

    #output data:
    fn = open(log_filename, 'a')
    fn.write('Accuracy Metrics: \n')
    dist = model_metrics_utils.get_accuracy_tplist(test, pred_id=pred_id)
    for key, acc in dist.items():
        fn.write('{}: {}\n'.format(key,numpy.mean(abs(numpy.array(acc)))))

    fn.write("\nMin/max for each keypoint\n")
    for key, acc in dist.items():
        fn.write('{}: {}, {}\n'.format(key, numpy.min(abs(numpy.array(acc))), numpy.max(abs(numpy.array(acc)))))

    worst_dist = model_metrics_utils.get_accuracy_tplist(test, pred_id='worst case keypoints')
    fn.write('\nWorst Case Metrics: \n')
    for key, acc in worst_dist.items():
        fn.write('{}: {}\n'.format(key,numpy.mean(abs(numpy.array(acc)))))

    fn.write("\nMin/max for each keypoint: \n")
    for key, acc in worst_dist.items():
        fn.write('{}: {}, {}\n'.format(key, numpy.min(abs(numpy.array(acc))), numpy.max(abs(numpy.array(acc)))))

    fn.write('\n\n\n')
    fn.close()

    #save predictions
    for experiment in experiments:
        experiment.write_to_disk()


if __name__ == "__main__":
    try:
        covariate, max_val = sys.argv[1], sys.argv[2]
    except IndexError:
        print("Please include covariate and max_val")
        sys.exit(1)

    project_name = '{}x{}_cov{}_max{}_test'.format(image_shape[0], image_shape[1], covariate, max_val)
    if mask_error:
        project_name+='_mask'
    if os_type == 'Darwin':
        model_path_root = '/Volumes/lugia_array/Laird_Nicolette/deep_learning/keypoint_detection/new_api/production_dataloader_test/new_kp_maps/gaussian_kp/'+project_name
    elif os_type == 'Linux':
        model_path_root = '/mnt/lugia_array/Laird_Nicolette/deep_learning/keypoint_detection/new_api/production_dataloader_test/new_kp_maps/gaussian_kp/'+project_name

    run_model_metrics(model_path_root, covariate, max_val)
