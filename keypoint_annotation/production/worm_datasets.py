from torch.utils import data
from zplib.image import colorize
from zplib.image import pyramid
from zplib.curve import interpolate
from zplib.curve import spline_geometry
from scipy import stats
import freeimage
import pickle
import numpy 
from elegant import process_images
from elegant import worm_spline
from elegant import datamodel

class GenerateWormImage:
    """Callable class that returns a worm-frame image when called with a Timepoint instance.
    This class can be used to generate 
    Shape of the worm-frame image can be configured at class initialization.


    """
    def __init__(self, downscale=2, image_shape=(960,512)):
        self.downscale = downscale

        self.WIDTH_TRENDS = pickle.load(open('/home/nicolette/.conda/envs/nicolette/lib/python3.7/site-packages/elegant/width_data/width_trends.pickle', 'rb'))
        self.AVG_WIDTHS = numpy.array([numpy.interp(5, self.WIDTH_TRENDS['ages'], wt) for wt in self.WIDTH_TRENDS['width_trends']])
        self.AVG_WIDTHS_TCK = self.to_tck(self.AVG_WIDTHS)
        self.image_shape = image_shape

    def __call__(self, timepoint):
        downscale = self.downscale
        bf = self.preprocess_image(timepoint)
        annotations = timepoint.annotations
        center_tck, width_tck = annotations['pose']

        image_size = (self.image_shape[0]/downscale, self.image_shape[1]/downscale)
        
        new_center_tck = (center_tck[0], center_tck[1]/downscale, center_tck[2])
        new_width_tck = (width_tck[0], width_tck[1]/downscale, width_tck[2])
        avg_widths = (self.AVG_WIDTHS_TCK[0], self.AVG_WIDTHS_TCK[1]/downscale, self.AVG_WIDTHS_TCK[2])
        
        reflect = False
        if 'keypoints' in annotations and 'vulva' in annotations['keypoints']:
            x, y = annotations['keypoints']['vulva']
            reflect = y < 0
        image_width, image_height = image_size
        worm_frame = worm_spline.to_worm_frame(bf, new_center_tck, new_width_tck,
            standard_width=avg_widths, zoom=1, order=1, sample_distance=image_height//2, standard_length=image_width, reflect_centerline=reflect)
        mask = worm_spline.worm_frame_mask(new_width_tck, worm_frame.shape)
        worm_frame[mask == 0] = 0
        return worm_frame

    @staticmethod
    def to_tck(widths):
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
        