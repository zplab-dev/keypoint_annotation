from torch.utils import data
from zplib.image import colorize
from zplib.image import pyramid
from zplib.curve import interpolate
from zplib.curve import spline_geometry
from scipy import stats
import freeimage
import pickle
import numpy 
import pkg_resources
from elegant import process_images
from elegant import worm_spline
from elegant import datamodel

class WormKeypointDataset:
    def __init__(self, generate_data, downscale=2, scale=(0,1,2,3), image_size=(960,512), limited=False):
        """Callable class to generate a worm frame image and scaled keypoint maps to be used in training the keypoint CNN.

        Parameters:
            generate_data: function or callablŽe class to generate keypoint maps using a tuple of x,y positions of the keypoints and
                the worm-frame image shape. Must have the signature (keypoint_coords, image_shape), where keypoint_coords is a tuple of x and y coords
                for the keypoints
            downscale: 
            scale:
            image_size:

        Returns:

        Example:
        """
        self.generate_keypoint_maps = generate_data
        self.downscale = downscale
        self.limited = limited

        with pkg_resources.resource_stream('elegant', 'width_data/width_trends.pickle') as f:
            trend_data = pickle.load(f)
            self.WIDTH_TRENDS = trend_data

        self.AVG_WIDTHS = numpy.array([numpy.interp(5, self.WIDTH_TRENDS['ages'], wt) for wt in self.WIDTH_TRENDS['width_trends']])
        AVG_WIDTHS_TCK = self.to_tck(self.AVG_WIDTHS)
        self.AVG_WIDTHS_TCK = (AVG_WIDTHS_TCK[0], AVG_WIDTHS_TCK[1]/downscale, AVG_WIDTHS_TCK[2])
        self.scale = scale
        self.image_size = image_size

    def __call__(self, timepoint):
        worm_frame_image = self.worm_frame_image(timepoint)
        #get keypoint maps
        keypoint_coords = self.get_keypoint_coords(timepoint, worm_frame_image.shape)
        keypoint_maps = self.generate_keypoint_maps(keypoint_coords, worm_frame_image.shape)
        #scale keypoint maps
        scaled_maps= self.scale_keypoint_maps(keypoint_maps, worm_frame_image.shape)
        #make worm frame image into a 3-D image
        extend_img = numpy.array([worm_frame_image,worm_frame_image,worm_frame_image])

        return extend_img, scaled_maps

    def to_tck(self, widths):
        x = numpy.linspace(0, 1, len(widths))
        smoothing = 0.0625 * len(widths)
        return interpolate.fit_nonparametric_spline(x, widths, smoothing=smoothing)

    @staticmethod
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

    def preprocess_image(self, timepoint):
        downscale = self.downscale
        lab_frame_image = freeimage.read(timepoint.image_path('bf'))
        lab_frame_image = lab_frame_image.astype(numpy.float32)
        height, width = lab_frame_image.shape[:2]

        try:
            metadata = timepoint.position.experiment.metadata
            optocoupler = metadata['optocoupler']
        except KeyError:
            optocoupler = 1
        mode = process_images.get_image_mode(lab_frame_image, optocoupler=optocoupler)

        #### DownSample the image 
        if downscale > 0 and downscale != 1:#and set_name!='train':        
            #t_size = (int(width / downscale), int(height / downscale))  
            shrink_image = pyramid.pyr_down(lab_frame_image, downscale=downscale)
            #shrink_image = numpy.clip(shrink_image, 0, 40000)   
        else:
            shrink_image = lab_frame_image

        shrink_image = shrink_image.astype(numpy.float32)

        ## scale the image pixel value into a trainable range
        # map image image intensities in range (100, 2*mode) to range (0, 2)
        bf = colorize.scale(shrink_image, min=100, max=2*mode, output_max=2)
        # now shift range to (-1, 1)
        bf -= 1
        return bf
        
    def worm_frame_image(self, timepoint):
        downscale = self.downscale
        bf = self.preprocess_image(timepoint)
        annotations = timepoint.annotations
        center_tck, width_tck = annotations['pose']
        image_size = self.image_size
        image_shape = (image_size[0]/downscale, image_size[1]/downscale)
        
        new_center_tck = (center_tck[0], center_tck[1]/downscale, center_tck[2])
        new_width_tck = (width_tck[0], width_tck[1]/downscale, width_tck[2])
        avg_widths = self.AVG_WIDTHS_TCK
        #avg_widths = (self.AVG_WIDTHS_TCK[0], self.AVG_WIDTHS_TCK[1]/downscale, self.AVG_WIDTHS_TCK[2])
        
        reflect = False
        if 'keypoints' in annotations and 'vulva' in annotations['keypoints']:
            x, y = annotations['keypoints']['vulva']
            reflect = y > 0
        image_width, image_height = image_shape
        worm_frame = worm_spline.to_worm_frame(bf, new_center_tck, new_width_tck,
            standard_width=avg_widths, zoom=1, order=1, sample_distance=image_height//2, standard_length=image_width, reflect_centerline=reflect)
        mask = worm_spline.worm_frame_mask(new_width_tck, worm_frame.shape)
        mask = mask>0
        worm_frame[~mask] = 0
        return worm_frame

    def get_keypoint_coords(self, timepoint, image_shape):
        #annotations = timepoint.annotations
        center_tck, width_tck = timepoint.annotations['pose']
        keypoints = timepoint.annotations['keypoints']

        #step 1: get the x,y positions in the new image shape
        length = spline_geometry.arc_length(center_tck)
        xs = numpy.array([keypoints[k][0] for k in ('anterior bulb', 'posterior bulb', 'vulva', 'tail')])
        x_percent = xs/length
        new_xs = x_percent*image_shape[0]

        #get y coordinates
        #put all keypoints except vulva at the midline
        ys = [int(image_shape[1]/2)]*len(new_xs)
        vulvax = int(new_xs[2])
        avg_widths = interpolate.spline_interpolate(self.AVG_WIDTHS_TCK, image_shape[0])
        vulvay = avg_widths[vulvax]
        #widths are wrt the midline, so put vulva on correct side
        ys[2] = (image_shape[1]/2) - vulvay

        return new_xs, ys


    def scale_keypoint_maps(self, keypoint_maps, image_size):
        scale_keypoint0_maps = []
        scale_keypoint1_maps = []
        scale_keypoint2_maps = []
        scale_keypoint3_maps = []

        for i in self.scale:
            s = 2 ** i
            xkp0 = pyramid.pyr_down(keypoint_maps[0], downscale=s)
            xkp0 = numpy.expand_dims(xkp0, axis=0)
            
            xkp1 = pyramid.pyr_down(keypoint_maps[1], downscale=s)
            xkp1 = numpy.expand_dims(xkp1, axis=0)

            xkp2 = pyramid.pyr_down(keypoint_maps[2], downscale=s)
            xkp2 = numpy.expand_dims(xkp2, axis=0)

            xkp3 = pyramid.pyr_down(keypoint_maps[3], downscale=s)
            xkp3 = numpy.expand_dims(xkp3, axis=0)
            
            scale_keypoint0_maps.append(xkp0.astype(numpy.float32))
            scale_keypoint1_maps.append(xkp1.astype(numpy.float32))
            scale_keypoint2_maps.append(xkp2.astype(numpy.float32))
            scale_keypoint3_maps.append(xkp3.astype(numpy.float32))

        if self.limited:
            return(scale_keypoint1_maps, scale_keypoint2_maps)
        else:
            return(scale_keypoint0_maps, scale_keypoint1_maps, scale_keypoint2_maps, scale_keypoint3_maps)

class GaussianKpMap2D:
    def __init__(self, covariate=100, max_val=100):
        self.covariate = covariate
        self.max_val = max_val

    def __call__(self, keypoint_coords, worm_frame_shape):
        covariate = self.covariate
        max_val = self.max_val

        #step 1: get the x,y positions in the new image shape
        xs, ys = keypoint_coords

        #step 2: make the gaussian hotspot over the x,y positions of the images
        keypoint_maps = []
        for x,y in zip(xs, ys):
            xidx, yidx = numpy.indices(worm_frame_shape)
            points = numpy.stack((xidx, yidx), axis=-1)
            pdf = stats.multivariate_normal.pdf(points, (x,y), covariate) #make gaussian pdf centered at x,y
            #kp_image = numpy.ones(worm_frame_shape)*pdf
            #scale image to make training easier
            kp_image = pdf/pdf.max() #get everything into the range 0,1
            kp_image *= max_val #to scale things to make loss better when training
            keypoint_maps.append(kp_image.astype(numpy.float32))

        return keypoint_maps

class GaussianKpMap1D:
    def __init__(self, covariate=100, max_val=100):
        self.covariate = covariate
        self.max_val = max_val

    def __call__(self, keypoint_coords, worm_frame_shape):
        covariate = self.covariate
        max_val = self.max_val

        #step 1: get the x,y positions in the new image shape
        xs, ys = keypoint_coords

        #step 2: make the gaussian hotspot over the x,y positions of the images
        keypoint_maps = []
        for x,y in zip(xs, ys):
            xidx, yidx = numpy.indices(worm_frame_shape)
            dev = numpy.sqrt(covariate) #scale is the std deviation, so need to take the sqrt of variance
            pdf = stats.norm.pdf(xidx, loc=x, scale=dev) #make 1D gaussian pdf centered at x with variance covariate 
            #kp_image = numpy.ones(worm_frame_shape)*pdf
            #scale image to make training easier
            kp_image = pdf/pdf.max() #get everything into the range 0,1
            kp_image *= max_val #to scale things to make loss better when training
            keypoint_maps.append(kp_image.astype(numpy.float32))

        return keypoint_maps

class SigmoidKpMap:
    def __init__(self, slope=1, max_val=1):
        self.slope = slope
        self.max_val = max_val

    def __call__(self, keypoint_coords, worm_frame_shape):
        slope = self.slope
        max_val = self.max_val

        #step 1: get the x,y positions in the new image shape
        xs, ys = keypoint_coords

        #step 2: make the gaussian hotspot over the x,y positions of the images
        keypoint_maps = []
        for x,y in zip(xs, ys):
            xcoords, ycoords = numpy.indices(worm_frame_shape)
            #subtract x point, so the keypoint is centered at zero
            xcoords = xcoords - x
            #apply sigmoid to xcoords
            kp_image = self.sigmoid(xcoords)
            #scale image to make training easier
            kp_image *= max_val #to scale things to make loss better when training
            keypoint_maps.append(kp_image.astype(numpy.float32))

        return keypoint_maps

    def sigmoid(self, x):
        slope = self.slope
        #we want the range to be (-1, 1)
        return -1+2/(1 + numpy.exp(-slope*x))
