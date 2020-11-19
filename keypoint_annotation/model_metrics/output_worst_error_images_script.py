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
            model_metrics_utils.predict_timepoint(timepoint, pred_id, model_paths, downscale, image_shape, sigmoid)
            model_metrics_utils.predict_worst_timepoint(timepoint, 'worst case keypoints', model_paths, downscale, image_shape, sigmoid)

    #output the worst error images
    save_dir = model_path_root+"/worst_images/"
    if not os.path.exists(save_dir): os.makedirs(save_dir)

    keypoint_list = ['anterior_bulb','posterior_bulb','vulva','tail', 'vulva_class']
    for i in range(4):
        save_name = save_dir+"{}_top5.png".format(keypoint_list[i])
        sorted_test = model_metrics_utils.sort_tp_list_by_error(test, i, pred_id=pred_id)
        plot_output_images(sorted_test, i, save_name, model_paths=model_paths, downscale=downscale, image_shape=image_shape, pred_id=pred_id)

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
    experiments = set([tp.position.experiment for tp in test])
    for experiment in experiments:
        experiment.write_to_disk()


def plot_output_images(timepoint_list, kp_idx, save_name, model_paths, downscale=1, image_shape=(960,96) , pred_id = 'pred keypoints'):
    plt.figure(figsize=(22,20), dpi=88, facecolor='w', edgecolor='k')
    fig, ax = plt.subplots(5,2, figsize=(15,30))
    for i in range(5):
        #axx = int(i/5)
        axx = i
        axy = 0
        #if i>4:
        #    axy = int(i-5)
        timepoint = timepoint_list[i]
        keypoint_list = ['anterior bulb','posterior bulb','vulva','tail', 'vulva class']
        kp = keypoint_list[kp_idx]
        keypoints = timepoint.annotations['keypoints']

        reflect = keypoints['vulva'][1] > 0
        tp_image = production_utils.get_worm_frame_image(timepoint, downscale=downscale, image_size=image_shape, reflect=reflect)
        extend_img = numpy.array([tp_image, tp_image, tp_image])
        gt_keypoints = timepoint.annotations['keypoints']
        output_images = production_utils.output_prediction_image(extend_img, gt_keypoints, model_paths=model_paths)
        
        pred_keypoints = timepoint.annotations[pred_id]
        
        d = model_metrics_utils.get_tp_accuracy(timepoint, pred_id=pred_id)
        
        pnorm_kp = production_utils.normalize_pred_keypoints(timepoint, pred_keypoints, downscale=downscale, image_size=image_shape)
        gtnorm_kp = production_utils.normalize_pred_keypoints(timepoint, gt_keypoints, downscale=downscale, image_size=image_shape)
        if kp == 'vulva class':
            gt_vulva = 0
            p_vulva = 0
            if gtnorm_kp['vulva'][1] >0:
                gt_vulva = 1
            if pnorm_kp['vulva'][1] >0:
                p_vulva = 1
        else:
            gtkp = gtnorm_kp[kp]
            pkp = pnorm_kp[kp]

        print("gt:", gtkp, "pkp:",pkp)
        circle = plt.Circle((pkp[1],pkp[0]), 3 , color='r')
        circle1 = plt.Circle((gtkp[1],gtkp[0]), 3 , color='cyan')
        ax[axx, axy].imshow(tp_image, cmap='gray')
        ax[axx, axy].add_patch(circle)
        ax[axx, axy].add_patch(circle1)
        circle.set(label='Pred kp')
        circle1.set(label='GT kp')
        
        ax[axx, 1].imshow(output_images[kp_idx])
        ax[axx, 1].axis('on')
        
        
        ax[axx,0].set_title(keypoint_list[kp_idx]+" error: "+str(d[kp]))
        ax[axx, 1].set_title('output from CNN')

        ax[axx,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)   

    plt.savefig(save_name)


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