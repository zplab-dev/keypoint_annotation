import itertools
import platform
from keypoint_annotation.model_metrics import model_metrics_script

def multi_model_metrics():
    #cov_par = [25, 50, 100, 200]
    os_type = platform.system()
    #model parameters
    cov_par = [25, 50, 200]
    #cov_par = [100]
    val_par = [1]
    slope =[0.25]
    #val_par = [3]
    downscale = 1
    image_shape = (960,96)
    mask_error = False
    sigmoid = True
    for covariate, max_val in itertools.product(slope, val_par):
        image_size = (int(image_shape[0]/downscale), int(image_shape[1]/downscale))
        project_name = '{}x{}_slope{}_max{}'.format(image_size[0], image_size[1], covariate, max_val)
        if mask_error:
            project_name+='_mask'
        if os_type == 'Darwin':
            model_path_root = '/Volumes/lugia_array/Laird_Nicolette/deep_learning/keypoint_detection/new_api/production_dataloader_test/new_kp_maps/sigmoid_kp/'+project_name
        elif os_type == 'Linux':
            model_path_root = '/mnt/lugia_array/Laird_Nicolette/deep_learning/keypoint_detection/new_api/production_dataloader_test/new_kp_maps/sigmoid_kp/'+project_name

        print("Running metrics for model: {}".format(project_name))
        model_metrics_script.run_model_metrics(model_path_root, covariate, max_val, downscale, image_shape, mask_error, sigmoid)

if __name__ == "__main__":
    multi_model_metrics()