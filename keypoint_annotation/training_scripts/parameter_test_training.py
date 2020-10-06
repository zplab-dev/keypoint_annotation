import itertools

from keypoint_annotation.training_scripts import gaussian_training_script
from keypoint_annotation.training_scripts import vulva_classifier_script

def parameter_test():
	#cov_par = [25, 50, 100, 200]
	cov_par = [100]
	val_par = [1, 3, 100]
	for covariate, max_val in itertools.product(cov_par, val_par):
		print("Training with covariate {}, max_val {}".format(covariate, max_val))
		save_dir = gaussian_training_script.train_model(covariate, max_val)
		vulva_classifier_script.train_vulva(save_dir)

if __name__ == "__main__":
    parameter_test()