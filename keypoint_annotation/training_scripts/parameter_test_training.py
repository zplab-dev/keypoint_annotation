import argparse
import itertools
import platform

from keypoint_annotation.training_scripts import gaussian_training_script
from keypoint_annotation.training_scripts import vulva_classifier_script
from keypoint_annotation.training_scripts import sigmoid_training_script


def parameter_test(sigmoid=False, dim1D=False, epochs=25):
    if sigmoid:
        cov_par =[0.01,  0.5,  1,  10]
    else:
        cov_par = [25, 50, 100, 200]
    #cov_par = [25, 200]
    #cov_par = [100]
    #val_par = [1,3,100]
    #slope = [0.25]
    val_par = [3]
    downscale = 1
    image_shape = (960,96)
    mask_error = False
    epochs = epochs

    for covariate, max_val in itertools.product(cov_par, val_par):
        print("Training with covariate {}, max_val {}".format(covariate, max_val))
        image_size = (int(image_shape[0]/downscale), int(image_shape[1]/downscale))
        if sigmoid:
            project_name = '{}x{}_slope{}_max{}_{}epochs'.format(image_size[0], image_size[1],covariate, max_val, epochs)
        else:
            project_name = '{}x{}_cov{}_max{}_{}epochs'.format(image_size[0], image_size[1],covariate, max_val, epochs)
        if mask_error:
            project_name += '_mask'
        #save_dir = './'+project_name
        if sigmoid:
            save_dir = sigmoid_training_script.train_model(covariate, max_val, downscale, image_shape, mask_error, epochs)
        else:
            save_dir = gaussian_training_script.train_model(covariate, max_val, downscale, image_shape, mask_error, epochs, dim1D)
        vulva_classifier_script.train_vulva(save_dir, downscale, image_shape)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sigmoid', default=False, action='store_true')
    parser.add_argument('--dim1D', default=False, action='store_true')
    parser.add_argument('--epochs', type=int, default=25)
    args = parser.parse_args()
    epochs = args.epochs

    if args.sigmoid:
        parameter_test(sigmoid=True, epochs=epochs)
    else:
        if args.dim1D:
            parameter_test(sigmoid=False, dim1D=True, epochs=epochs)
        else:
            parameter_test(sigmoid=False, dim1D=False, epochs=epochs)