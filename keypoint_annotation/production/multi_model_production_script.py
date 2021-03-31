import itertools
import platform
import argparse
from keypoint_annotation.production import production_script

def multi_model_predictions(experiment, sigmoid=False, epochs=None):
    os_type = platform.system()
    if os_type == 'Darwin':
            model_path_root = '/Volumes/squidarray/Laird_Nicolette/keypoint_annotations/'
    else:
        model_path_root = '/mnt/squidarray/Laird_Nicolette/keypoint_annotations/'

    if sigmoid:
            model_path_root += 'sigmoid_kp/'
    elif dim1D:
        model_path_root+='1D_gaussian/'
    else:
        model_path_root+='2D_gaussian/'

    #model parameters
    if sigmoid:
        cov_par =[0.01, 0.25, 0.5,  1,  10]
    else:
        cov_par = [25, 50, 100, 200]
    
    val_par=[3]
    downscale = 1
    image_shape = (960,96)
    mask_error = False
    for covariate, max_val in itertools.product(cov_par, val_par):
        image_size = (int(image_shape[0]/downscale), int(image_shape[1]/downscale))
        if sigmoid:
            project_name = '{}x{}_slope{}_max{}'.format(image_size[0], image_size[1], covariate, max_val)
        else:
            project_name = '{}x{}_cov{}_max{}'.format(image_size[0], image_size[1], covariate, max_val)

        if epochs:
            project_name = str(epochs)+'_epochs/'+project_name
        if mask_error:
            project_name+='_mask'

        
        root_path= model_path_root+project_name

        print("Running metrics for model: {}".format(project_name))
        model_metrics_script.run_model_metrics(root_path, covariate, max_val, downscale, image_shape, mask_error, sigmoid, epochs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sigmoid', default=False, action='store_true')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('exp_root', action='store', type=str)
    args = parser.parse_args()
    epochs = None
    if args.epochs:
        epochs = args.epochs
        print(epochs)
    experiment_root = args.exp_root
    experiment = datamodel.Experiment(experiment_root)

    if args.sigmoid:
        multi_model_predictions(experiment, sigmoid=True, epochs=epochs)
    else:
        
        multi_model_metrics(experiment, sigmoid=False, epochs=epochs)