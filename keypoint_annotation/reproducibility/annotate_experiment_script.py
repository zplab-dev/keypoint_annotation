import pathlib
import platform
import sys

from ris_widget import ris_widget
from elegant import process_images
from elegant import worm_spline
from elegant import datamodel
from elegant.gui import pose_annotation 
from elegant.gui import keypoint_annotation 
from elegant.gui import experiment_annotator


def set_up_annotations(initials, os):
    initials = initials
    timepoint_path, new_root = None, None
    if os == 'Darwin':
        #timepoint_path = '/Volumes/lugia_array/Laird_Nicolette/deep_learning/keypoint_detection/new_api/production_dataloader_test/reproducibility/sampled_timepoints_os.txt'
        timepoint_path = '/Volumes/lugia_array/Laird_Nicolette/deep_learning/keypoint_detection/new_api/production_dataloader_test/training_paths/test_path_os.txt'
        new_root = pathlib.Path('/Volumes/lugia_array/Laird_Nicolette/deep_learning/keypoint_detection/new_api/production_dataloader_test/reproducibility/test_data/')
    elif os == 'Linux':
        #timepoint_path = '/mnt/lugia_array/Laird_Nicolette/deep_learning/keypoint_detection/new_api/production_dataloader_test/reproducibility/sampled_timepoints_linux.txt'
        timepoint_path = '/mnt/lugia_array/Laird_Nicolette/deep_learning/keypoint_detection/new_api/production_dataloader_test/training_paths/test_path_linux.txt'
        new_root = pathlib.Path('/mnt/lugia_array/Laird_Nicolette/deep_learning/keypoint_detection/new_api/production_dataloader_test/reproducibility/test_data/')
    elif os == 'Windows':
        #assume they are running things from mobaxterm (not the best idea, but it's what I have)
        #0timepoint_path = '/mnt/lugia_array/Laird_Nicolette/deep_learning/keypoint_detection/new_api/production_dataloader_test/reproducibility/sampled_timepoints_linux.txt'
        timepoint_path = '/mnt/lugia_array/Laird_Nicolette/deep_learning/keypoint_detection/new_api/production_dataloader_test/training_paths/test_path_linux.txt'
        new_root = pathlib.Path('/mnt/lugia_array/Laird_Nicolette/deep_learning/keypoint_detection/new_api/production_dataloader_test/reproducibility/test_data/')

    sample_timepoints = datamodel.Timepoints.from_file(timepoint_path)
    
    for tp in sample_timepoints:
        position = tp.position
        position.relocate_annotations(new_root / position.experiment.name, copy_original=False)

    class VirtualPosition:
        def __init__(self, name, timepoints):
            self.name = name
            self.timepoints = timepoints
            self.annotations = {}
        def __len__(self):
            return len(self.timepoints)
        def write_annotations(self):
            for position in set(timepoint.position for timepoint in self.timepoints):
                position.write_annotations()
        def __iter__(self):
            return iter(self.timepoints)

    position = VirtualPosition('reproducibility',sample_timepoints)


    rw = ris_widget.RisWidget()
    pa = pose_annotation.PoseAnnotation(rw)
    keypoint_name = 'keypoints '+initials
    ka = keypoint_annotation.KeypointAnnotation(rw, ['anterior bulb', 'posterior bulb', 'vulva', 'tail'], name= keypoint_name, auto_advance=True)
    ea = experiment_annotator.ExperimentAnnotator(rw, [position], [pa, ka])
    rw.run()


if __name__ == "__main__":
    try:
        initials = str(sys.argv[1])
        os = platform.system()
    except IndexError:
        print("No initials found")
        sys.exit(1)

    set_up_annotations(initials, os)



