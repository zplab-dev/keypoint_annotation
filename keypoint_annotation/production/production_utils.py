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

def get_metadata(timepoint):
    metadata = timepoint.position.experiment.metadata
    try:
        objective, optocoupler, temp = metadata['objective'], metadata['optocoupler'], metadata['nominal_temperature']
    except KeyError:
        objective = 5
        optocoupler = 1
        temp = 25
    magnification = objective * optocoupler
    return objective, optocoupler, magnification, temp

def preprocess_image(timepoint, downscale):
    img_path = timepoint.image_path('bf')
    lab_frame_image = freeimage.read(img_path)
    lab_frame_image = lab_frame_image.astype(numpy.float32)
    height, width = lab_frame_image.shape[:2]
    objective, optocoupler, magnification, temp = get_metadata(timepoint)

    mode = process_images.get_image_mode(lab_frame_image, optocoupler=optocoupler)
    #### DownSample the image 
    if downscale > 0 and downscale != 1:#and set_name!='train':        
        shrink_image = pyramid.pyr_down(lab_frame_image, downscale=downscale)
    else:
        shrink_image = lab_frame_image
    shrink_image = shrink_image.astype(numpy.float32)
    ## scale the image pixel value into a trainable range
    # map image image intensities in range (100, 2*mode) to range (0, 2)
    bf = colorize.scale(shrink_image, min=100, max=2*mode, output_max=2)
    # now shift range to (-1, 1)
    bf -= 1
    return bf

def get_worm_frame_image(timepoint, downscale=1, image_size=(960, 512), reflect=False):
    bf = preprocess_image(timepoint, downscale)
    annotations = timepoint.annotations
    center_tck, width_tck = annotations['pose']
    image_shape = (image_size[0]/downscale, image_size[1]/downscale)
    #deal with downscaling
    new_center_tck = (center_tck[0], center_tck[1]/downscale, center_tck[2])
    new_width_tck = (width_tck[0], width_tck[1]/downscale, width_tck[2])
    avg_widths = (AVG_WIDTHS_TCK[0], AVG_WIDTHS_TCK[1]/downscale, AVG_WIDTHS_TCK[2])

    reflect = reflect

    image_width, image_height = image_shape
    worm_frame = worm_spline.to_worm_frame(bf, new_center_tck, new_width_tck,
        standard_width=avg_widths, zoom=1, order=1, sample_distance=image_height//2, standard_length=image_width, reflect_centerline=reflect)
    mask = worm_spline.worm_frame_mask(avg_widths, worm_frame.shape)
    worm_frame[mask == 0] = 0
    return worm_frame

def weighted_mean(kp_map):
	#try a gaussian filter to make output a bit smoother
    out_kp_map = gaussian_filter(kp_map, 1)
    #To ensure there are no negative values, make the pixels that are less than half the maximum zero
    max_pixel = out_kp_map.max()
    out_kp_map[out_kp_map<(max_pixel/2)] = 0
    x, y = numpy.indices(out_kp_map.shape)
    pred_x = (x * out_kp_map).sum() / out_kp_map.sum()
    pred_y = (y * out_kp_map).sum() / out_kp_map.sum() 
    print("pred_x, pred_y: ", pred_x, pred_y)
    return (pred_x, pred_y)

def process_sigmoid(out_kp_map):
	max_val = out_kp_map.max() #can have a variable value as the max value
	return max_val - abs(out_kp_map)

def process_output(out, downscale=2, sigmoid=False):
    #Way to get the keypoint maps and make it into the xy positions
    out_kp_map = out[('Keypoint0',0)][0].cpu().detach().numpy()
    out_kp_map = out_kp_map[0].copy()
    image_shape = out_kp_map.shape
    widths_tck = (AVG_WIDTHS_TCK[0], AVG_WIDTHS_TCK[1]/downscale, AVG_WIDTHS_TCK[2])
    #mask = worm_spline.worm_frame_mask(widths_tck, image_shape) #make worm mask
    #mask = mask>0
    #See if the output is a sigmoid or gaussian output type
    if sigmoid:
    	out_kp_map = process_sigmoid(out_kp_map)
    #if not sigmoid, assume the output is a guassian keypoint map, so go through with the weighted mean
    pred_x, pred_y = weighted_mean(out_kp_map)
    return (pred_x, pred_y)


def renormalize_pred_keypoints(timepoint, pred_keypoints, downscale=2, image_size=(960,512)):
    downscale = downscale
    print('timepoint: ', timepoint.position.experiment.name, timepoint.position.name, timepoint.name)
    center_tck, width_tck = timepoint.annotations['pose']
    image_shape = (image_size[0]/downscale, image_size[1]/downscale)
    length = spline_geometry.arc_length(center_tck)
    sample_dist = interpolate.spline_interpolate(width_tck, int(length)).max()+20
    width = int(round(sample_dist*2))
    new_keypoints = {}
    for kp, points in pred_keypoints.items():
        x,y = points

        x_percent = x/image_shape[0]
        new_x = x_percent*length
        if kp is 'vulva':
            vulvax = int(new_x)
            print("pred_keypoints: ", x, y)
            print("vulvax: ", vulvax)
            print("x_percent: ", x_percent)
            avg_widths = interpolate.spline_interpolate(width_tck, length)
            if vulvax >= len(avg_widths):
                vulvax = len(avg_widths)-1
            vulvay = avg_widths[vulvax]
            if y <0:
                new_y = -vulvay
            else:
                new_y = vulvay
        else:
            new_y=0

        new_keypoints[kp] = (new_x, new_y)

    return new_keypoints

def normalize_pred_keypoints(timepoint, pred_keypoints, downscale=2, image_size=(960,512)):
    downscale = downscale
    center_tck, width_tck = timepoint.annotations['pose']
    new_width_tck = (AVG_WIDTHS_TCK[0], AVG_WIDTHS_TCK[1]/downscale, AVG_WIDTHS_TCK[2])
    image_shape = (image_size[0]/downscale, image_size[1]/downscale)
    length = spline_geometry.arc_length(center_tck)
    sample_dist = interpolate.spline_interpolate(width_tck, length).max()+20
    if pred_keypoints is None:
        print("Keypoints do not exist with id: ", pred_id)
        return None
    new_keypoints = {}
    for kp, points in pred_keypoints.items():
        x,y = points
        x_percent = x/length
        new_x = x_percent*image_shape[0]
        new_y = int(image_shape[1]/2)
        if kp == 'vulva':
            vulvax = int(new_x)
            avg_widths = interpolate.spline_interpolate(new_width_tck, image_shape[0])
            if vulvax == len(avg_widths):
                vulvax = vulvax-1
            vulvay = avg_widths[vulvax]
            if y <0:
                new_y = (image_shape[1]/2) - vulvay
            else:
                new_y = (image_shape[1]/2) + vulvay
            

        new_keypoints[kp] = (new_x, new_y)

    return new_keypoints

def production_predict_image(image, keypoints, downscale=2, model_paths= {'ant_pharynx':"./models/ant_pharynx_bestValModel.paramOnly", 'post_pharynx':'./models/post_pharynx_bestValModel.paramOnly', 'vulva_class':'./models/vulva_class_bestValModel.paramOnly',
        'vulva_kp':'./models/vulva_kp_flip_bestValModel.paramOnly', 'tail':'./models/tail_bestValModel.paramOnly'}):
    
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
    """if keypoints['vulva'][1] > 0:
                    tensor_img = torch.flip(tensor_img, [3])"""
    #print(tensor_img.size())
    print("true vulva: ", keypoints['vulva'][1], "    vulva out: ", out, vulva_out)

    for kp, model_kp in zip(['anterior bulb', 'posterior bulb', 'vulva', 'tail'], ['ant_pharynx', 'post_pharynx', 'vulva_kp', 'tail']):
        #load model
        #print("Loading model: ", model_paths[model_kp])
        regModel = keypoint_annotation_model.WormRegModel(34, SCALE, pretrained=True)
        regModel.load_state_dict(torch.load(model_paths[model_kp], map_location='cpu'))
        regModel.eval()

        tensor_img = torch.tensor(image).unsqueeze(0)
        out = regModel(tensor_img)
        pred_kp = process_reg_output(out, downscale)
        if kp is 'vulva':
            #want to preserve what side the vulva is on.
            if vulva_out <=0:
                x,y = pred_kp
                pred_kp = (x, -y)
        keypoint_dict[kp] = pred_kp

    return keypoint_dict

def output_prediction_image(image, keypoints, model_paths={'ant_pharynx':"./models/ant_pharynx_bestValModel.paramOnly", 'post_pharynx':'./models/post_pharynx_bestValModel.paramOnly', 'vulva_class':'./models/vulva_class_bestValModel.paramOnly',
        'vulva_kp':'./models/vulva_kp_flip_bestValModel.paramOnly', 'tail':'./models/tail_bestValModel.paramOnly'}):
    keypoint_maps = []
    tensor_img = torch.tensor(image).unsqueeze(0)
    #flip vulva 
    """if keypoints['vulva'][1] > 0:
                    tensor_img = torch.flip(tensor_img, [3])"""
    #print(tensor_img.size())

    for kp, model_kp in zip(['anterior bulb', 'posterior bulb', 'vulva', 'tail'], ['ant_pharynx', 'post_pharynx', 'vulva_kp', 'tail']):
        #load model
        #print("Loading model: ", model_paths[model_kp])
        regModel = keypoint_annotation_model.WormRegModel(34, SCALE, pretrained=True)
        regModel.load_state_dict(torch.load(model_paths[model_kp], map_location='cpu'))
        regModel.eval()

        tensor_img = torch.tensor(image).unsqueeze(0)
        out = regModel(tensor_img)
        out_kp_map = out[('Keypoint0',0)][0].cpu().detach().numpy()
        out_kp_map = out_kp_map[0]
        keypoint_maps.append(out_kp_map)

    return keypoint_maps