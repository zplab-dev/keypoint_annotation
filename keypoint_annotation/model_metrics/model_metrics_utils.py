import pkg_resources
import freeimage
import numpy
import pickle
import torch

from scipy.ndimage import gaussian_filter

from torch.utils import data
from zplib.image import colorize
from zplib.image import pyramid
from zplib.curve import spline_geometry
from zplib.curve import interpolate

from elegant import process_images
from elegant import worm_widths
from elegant import worm_spline
from elegant import datamodel

from keypoint_annotation import keypoint_annotation_model

from keypoint_annotation.production import production_utils

def to_tck(widths):
    x = numpy.linspace(0, 1, len(widths))
    smoothing = 0.0625 * len(widths)
    return interpolate.fit_nonparametric_spline(x, widths, smoothing=smoothing)

with pkg_resources.resource_stream('elegant', 'width_data/width_trends.pickle') as f:
    trend_data = pickle.load(f)
    WIDTH_TRENDS = trend_data

AVG_WIDTHS = numpy.array([numpy.interp(5, WIDTH_TRENDS['ages'], wt) for wt in WIDTH_TRENDS['width_trends']])
AVG_WIDTHS_TCK = to_tck(AVG_WIDTHS)
SCALE = [0,1,2,3]

def predict_timepoint_list(timepoint_list, pred_id = 'pred keypoints test', model_paths= {'ant_pharynx':"./models/ant_pharynx_bestValModel.paramOnly", 'post_pharynx':'./models/post_pharynx_bestValModel.paramOnly', 
        'vulva_class':'./models/vulva_class_bestValModel.paramOnly','vuvla_kp':'./models/vulva_kp_flip_bestValModel.paramOnly', 'tail':'./models/tail_bestValModel.paramOnly'}, 
        downscale=2, image_shape=(960,512), sigmoid=False):
    """Convience function to predict all timepoints in a timepoint list
    """
    for timepoint in timepoint_list:
        predict_timepoint(timepoint, pred_id, model_paths, downscale, image_shape, sigmoid)

def predict_experiment(experiment, pred_id = 'pred keypoints test', model_paths= {'ant_pharynx':"./models/ant_pharynx_bestValModel.paramOnly", 'post_pharynx':'./models/post_pharynx_bestValModel.paramOnly', 
        'vulva_class':'./models/vulva_class_bestValModel.paramOnly','vuvla_kp':'./models/vulva_kp_flip_bestValModel.paramOnly', 'tail':'./models/tail_bestValModel.paramOnly'}, 
        downscale=2, image_shape=(960,512), sigmoid=False):
    """Convience function to predict all timepoints in an Experiment instance
    """
    for position in experiment.positions.values():
        for timepoint in position.timepoints.values():
            print(position.name, timepoint.name)
            predict_timepoint(timepoint, pred_id, model_paths, downscale, image_shape, sigmoid)

def predict_position(position, pred_id = 'pred keypoints test', model_paths= {'ant_pharynx':"./models/ant_pharynx_bestValModel.paramOnly", 'post_pharynx':'./models/post_pharynx_bestValModel.paramOnly', 
        'vulva_class':'./models/vulva_class_bestValModel.paramOnly','vuvla_kp':'./models/vulva_kp_flip_bestValModel.paramOnly', 'tail':'./models/tail_bestValModel.paramOnly'}, 
        downscale=2, image_shape=(960,512), sigmoid=False):
    """Convience function to predict all timepoints in Position instance
    """
    for timepoint in position.timepoints.values():
        predict_timepoint(timepoint, pred_id, model_paths, downscale, image_shape, sigmoid)

def predict_timepoint(timepoint, pred_id = 'pred keypoints test', model_paths= {'ant_pharynx':"./models/ant_pharynx_bestValModel.paramOnly", 'post_pharynx':'./models/post_pharynx_bestValModel.paramOnly', 
        'vulva_class':'./models/vulva_class_bestValModel.paramOnly','vuvla_kp':'./models/vulva_kp_flip_bestValModel.paramOnly', 'tail':'./models/tail_bestValModel.paramOnly'}, 
        downscale=2, image_shape=(960,512), sigmoid=False):
    
    #get worm-frame image
    worm_frame_image = production_utils.get_worm_frame_image(timepoint, downscale, image_shape, reflect=False)
    """if keypoints['vulva'][1] > 0:
                        worm_frame_image = numpy.flip(worm_frame_image, axis=1)"""
    worm_frame_image = numpy.expand_dims(worm_frame_image, axis=0)
    extend_img = numpy.concatenate((worm_frame_image, worm_frame_image, worm_frame_image),axis=0)
    #predict image and renormalize keypoints to the original image size
    keypoints = timepoint.annotations['keypoints']
    pred_keypoints = predict_image(extend_img, keypoints, downscale, model_paths, sigmoid)
    
    keypoint_dict = production_utils.renormalize_pred_keypoints(timepoint, pred_keypoints, downscale, image_shape)
    timepoint.annotations[pred_id] = keypoint_dict
    return keypoint_dict

def predict_image(image, keypoints, downscale=2, model_paths= {'ant_pharynx':"./models/ant_pharynx_bestValModel.paramOnly", 'post_pharynx':'./models/post_pharynx_bestValModel.paramOnly', 'vulva_class':'./models/vulva_class_bestValModel.paramOnly',
        'vulva_kp':'./models/vulva_kp_flip_bestValModel.paramOnly', 'tail':'./models/tail_bestValModel.paramOnly'}, sigmoid=False):
    keypoint_dict = {}
    #identify dorsal, ventral first
    regModel = keypoint_annotation_model.init_vuvla_class_model()
    regModel.load_state_dict(torch.load(model_paths['vulva_class'], map_location='cpu'))
    regModel.eval()
    tensor_img = torch.tensor(image).unsqueeze(0)
    out= regModel(tensor_img)
    _, vulva_out = torch.max(out, 1)
    vulva_out = vulva_out.item()
    #flip vulva if needed
    if keypoints['vulva'][1] > 0:
        tensor_img = torch.flip(tensor_img, [3])
    #print(tensor_img.size())
    print("true vulva: ", keypoints['vulva'][1], "    vulva out: ", out, vulva_out)

    for kp, model_kp in zip(['anterior bulb', 'posterior bulb', 'vulva', 'tail'], ['ant_pharynx', 'post_pharynx', 'vulva_kp', 'tail']):
        #load model
        #print("Loading model: ", model_paths[model_kp])
        regModel = keypoint_annotation_model.WormRegModel(34, SCALE, pretrained=True)
        regModel.load_state_dict(torch.load(model_paths[model_kp], map_location='cpu'))
        regModel.eval()
        
        out = regModel(tensor_img)
        pred_kp = production_utils.process_output(out, downscale, sigmoid)
        if kp is 'vulva':
            #want to preserve what side the vulva is on.
            if vulva_out <=0:
                x,y = pred_kp
                pred_kp = (x, -y)
        keypoint_dict[kp] = pred_kp

    return keypoint_dict

### Worst case prediction
def predict_worst_timepoint(timepoint, pred_id = 'worst case keypoints', model_paths= {'ant_pharynx':"./models/ant_pharynx_bestValModel.paramOnly", 'post_pharynx':'./models/post_pharynx_bestValModel.paramOnly', 
        'vulva_class':'./models/vulva_class_bestValModel.paramOnly','vuvla_kp':'./models/vulva_kp_flip_bestValModel.paramOnly', 'tail':'./models/tail_bestValModel.paramOnly'}, 
        downscale=2, image_shape=(960,512), sigmoid=False):
    
    #get worm-frame image
    #flip everything so that the vulva will be on the wrong side every time
    worm_frame_image = production_utils.get_worm_frame_image(timepoint, downscale, image_shape, reflect=True)
    keypoints = timepoint.annotations['keypoints']
    extend_img = numpy.array([worm_frame_image, worm_frame_image, worm_frame_image])
    #extend_img = numpy.concatenate((worm_frame_image, worm_frame_image, worm_frame_image),axis=0)
    #predict image and renormalize keypoints to the original image size
    keypoints = timepoint.annotations['keypoints']
    pred_keypoints = predict_image(extend_img, keypoints, downscale, model_paths, sigmoid)
    
    keypoint_dict = production_utils.renormalize_pred_keypoints(timepoint, pred_keypoints, downscale, image_shape)
    timepoint.annotations[pred_id] = keypoint_dict
    return keypoint_dict

######### Analytics functions ##########
def get_tp_accuracy(timepoint, pred_id='pred keypoints test'):
    dist = {}
    gt_kp = timepoint.annotations.get('keypoints', None)
    pose = timepoint.annotations.get('pose', None)
    pred_kp = timepoint.annotations.get(pred_id)
    if gt_kp is None or pred_kp is None or None in gt_kp.values() or None in pred_kp.values():
        print("None found in keypoint")
        return
    elif False in [x in list(gt_kp.keys()) for x in ['anterior bulb','posterior bulb','vulva','tail']]: 
        return
    else:
        for key in pred_kp.keys():
            gtx, gty = gt_kp[key]
            px, py = pred_kp[key]
            if key == 'vulva':
                dist['vulva'] = gtx - px
                dist['vulva class'] = ((gty* py) >= 0) #test sign
            else:
                dist[key] = gtx-px
            

    return dist

def get_accuracy_tplist(timepoint_list, pred_id='pred keypoints test'):
    dist = {'anterior bulb':[], 'posterior bulb':[], 'vulva':[],'vulva class':[], 'tail':[], 'age':[]}
    for timepoint in timepoint_list:
        acc = get_tp_accuracy(timepoint, pred_id)
        age = calculate_tp_age(timepoint)
        #print(acc.keys())
        for kp in acc.keys():
            #print(kp)
            dist[kp] += [acc[kp]]
        dist['age'] += [age]
    return dist

def get_accuracy(experiment, pred_id='pred keypoints test'):
    dist = {'anterior bulb':[], 'posterior bulb':[], 'vulva':[],'vulva class':[], 'tail':[]}

    for worm_name, timepoints in experiment.positions.items():
        for timepoint in timepoints.values():
            gt_kp = timepoint.annotations.get('keypoints')
            pose = timepoint.annotations.get('pose')
            pred_kp = timepoint.annotations.get(pred_id)
            if gt_kp is None or pred_kp is None:
                continue
            if None in gt_kp.values() or None in pred_kp.values():
                continue
            elif False in [x in list(gt_kp.keys()) for x in ['anterior bulb','posterior bulb','vulva','tail']]: 
                continue
            else:
                print(worm_name, timepoint)
                #true_kp = normalize_keypoints(pose, gt_kp, 2)
                for key in pred_kp.keys():
                    print(key)
                    gtx, gty = true_kp[key]
                    px, py = pred_kp[key]
                    if key is 'vulva':
                        dist['vulva']+=[gtx - px]
                        dist['vulva class'] = dist['vulva class']+[((gty* py) >= 0)] #test sign
                        print(key)
                        print("gt: ", gt_kp[key])
                        print("pred: ", pred_kp[key])
                        print("dist: ", str(gtx - px))
                    else:
                        dist[key] = dist[key]+[gtx-px]
            print("dist len: ",len(dist['vulva']))

    error = {key: np.mean(abs(np.array(vals))) for key, vals in dist.items()}
    return dist, error

def calculate_tp_age(timepoint):
    hatch_timestamp = timepoint.position.annotations.get('hatch_timestamp', None)
    if hatch_timestamp is None:
        #assume the first timepoint is the first larval position
        hatch_timestamp = next(iter(timepoint.position.timepoints.values())).annotations['timestamp']
    
    hours = (timepoint.annotations['timestamp']-hatch_timestamp)/3600

    return hours

def get_accuracy_and_age(experiment, pred_id = 'pred keypoints'):
    dist = {'anterior bulb':[], 'posterior bulb':[], 'vulva':[],'vulva class':[], 'tail':[], 'age': []}    
    for worm_name, positions in experiment.positions.items():
        for timepoint in positions.timepoints.values():
            acc = get_tp_accuracy(timepoint, pred_id)
            age = calculate_tp_age(timepoint)
            #print(acc.keys())
            for kp in acc.keys():
                #print(kp)
                dist[kp] += [acc[kp]]
            dist['age'] += [age]

    return dist

def sort_tp_list_by_error(tp_list, kp_idx, pred_id = 'pred keypoints'):
    """Note that this only works if you've run the predict_timepoint_list function already"""
    keypoint_list = ['anterior bulb','posterior bulb','vulva','tail', 'vulva class']

    def get_kp_accuracy(timepoint, keypoint=keypoint_list[kp_idx], pred_id= pred_id):
        acc = get_tp_accuracy(timepoint, pred_id)
        dist = acc[keypoint]
        return abs(dist)
        """gt_kp = timepoint.annotations.get('keypoints', None)
                                pose = timepoint.annotations.get('pose', None)
                                pred_kp = timepoint.annotations.get(pred_id)
                                dist = 0
                                if gt_kp is None or pred_kp is None or None in gt_kp.values() or None in pred_kp.values():
                                    print("None found in keypoint")
                                    return 0
                                elif False in [x in list(gt_kp.keys()) for x in ['anterior bulb','posterior bulb','vulva','tail']]: 
                                    return
                                else:
                                    gtx, gty = gt_kp[keypoint]
                                    px, py = pred_kp[keypoint]
                                    if keypoint == 'vulva class':
                                        dist = ((gty* py) >= 0) #test sign
                                    else:
                                        dist = gtx-px
                                print(timepoint.position.experiment.name,timepoint.position.name, timepoint.name, dist)
                                        
                                return abs(dist)"""

    return sorted(tp_list, key=get_kp_accuracy, reverse=True)
