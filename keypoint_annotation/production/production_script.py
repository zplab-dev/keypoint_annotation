import itertools
import platform
import argparse
from elegant import datamodel

from keypoint_annotation.model_metrics import model_metrics_script
from keypoint_annotation.production import production_utils
from keypoint_annotation.model_metrics import model_metrics_utils

def run_predictor(experiment, model_path_root, covariate, max_val, downscale=1, image_shape=(960,96), mask_error=False, sigmoid=False, dim1D=False):
	device ='cpu'
	if torch.cuda.is_available(): device='cuda:0'
    timepoint_list = datamodel.Timepoints.from_experiments(experiment) 

    device ='cpu'
    if torch.cuda.is_available(): device='cuda:0'
    image_size = (int(image_shape[0]/downscale), int(image_shape[1]/downscale))
    if sigmoid:
        pred_id = 'pred keypoints {}x{}_slope{}_max{}'.format(image_size[0], image_size[1], covariate, max_val)
    else:
        pred_id = 'pred keypoints {}x{}_cov{}_max{}'.format(image_size[0], image_size[1], covariate, max_val)
    
    if 'epochs' in model_path_root:
        m = re.search("\w+epochs", model_path_root)
        epoch_num = m.group()
        pred_id+='_'+epoch_num
    if mask_error:
        pred_id+='_mask'

    limited = False
    for timepoint in timepoint_list:
            model_metrics_utils.predict_timepoint(timepoint, pred_id, model_paths, downscale, image_shape, sigmoid, limited)

    experiment.write_to_disk() 

if __name__ == "__main__":
    os_type = platform.system()
    parser = argparse.ArgumentParser()
    parser.add_argument('--sigmoid', default=False, action='store_true')
    parser.add_argument('--dim1D', default=False, action='store_true')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('exp_root', action='store', type=str)
    args = parser.parse_args()
    epochs = None
    if args.epochs:
        epochs = args.epochs
        print(epochs)

    sigmoid = False
    if args.sigmoid:
        sigmoid = True

    #model parameters
    downscale = 1
    image_shape = (960,96)
    mask_error = False
    print(os_type)
    covariate = 200
    max_val = 3
    """try:
                    covariate, max_val, experiment_root = sys.argv[1], sys.argv[2], sys.argv[3]
                except IndexError:
                    print("Please include covariate and max_val")
                    sys.exit(1)"""

    image_size = (int(image_shape[0]/downscale), int(image_shape[1]/downscale))
    #project_name = '{}x{}_cov{}_max{}'.format(image_size[0], image_size[1], covariate, max_val)
    if sigmoid:
    	covariate = 1
    	project_name = '{}x{}_slope{}_max{}'.format(image_size[0], image_size[1], covariate, max_val)
    else:
    	project_name = '{}x{}_cov{}_max{}'.format(image_size[0], image_size[1], covariate, max_val)
    if mask_error:
        project_name+='_mask'
    if os_type == 'Darwin':
        model_path_root = '/Volumes/lugia_array/Laird_Nicolette/deep_learning/keypoint_detection/new_api/production_dataloader_test/new_kp_maps/'
        #model_path_root = '/Volumes/lugia_array/Laird_Nicolette/deep_learning/keypoint_detection/new_api/production_dataloader_test/new_api_960x96_cov100'
    elif os_type == 'Linux':
       #model_path_root = '/mnt/lugia_array/Laird_Nicolette/deep_learning/keypoint_detection/new_api/production_dataloader_test/new_api_960x96_cov100'
       model_path_root = '/Volumes/lugia_array/Laird_Nicolette/deep_learning/keypoint_detection/new_api/production_dataloader_test/new_kp_maps/'

    experiment = datamodel.Experiment(experiment_root)
    run_predictor(experiment, model_path_root, covariate, max_val, downscale, image_shape, mask_error, sigmoid, limited=False)