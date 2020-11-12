import itertools
import platform

from keypoint_annotation.training_scripts import gaussian_training_script
from keypoint_annotation.training_scripts import vulva_classifier_script
from keypoint_annotation.training_scripts import sigmoid_training_script

def parameter_test():
    cov_par = [25, 50, 100, 200]
    #cov_par = [25, 200]
    #cov_par = [100]
    #val_par = [1,3,100]
    #slope = [0.25]
    #slope =[0.01,  0.5,  1,  10]
    val_par = [3]
    downscale = 1
    image_shape = (960,96)
    mask_error = False

    for covariate, max_val in itertools.product(cov_par, val_par):
        print("Training with covariate {}, max_val {}".format(covariate, max_val))
        image_size = (int(image_shape[0]/downscale), int(image_shape[1]/downscale))
        project_name = '{}x{}_cov{}_max{}'.format(image_size[0], image_size[1],covariate, max_val)
        if mask_error:
            project_name += '_mask'
        #save_dir = './'+project_name
        save_dir = gaussian_training_script.train_model(covariate, max_val, downscale, image_shape, mask_error)
        vulva_classifier_script.train_vulva(save_dir, downscale, image_shape)

if __name__ == "__main__":
    parameter_test()