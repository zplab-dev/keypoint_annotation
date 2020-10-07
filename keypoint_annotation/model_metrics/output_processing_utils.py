from torch.utils import data
from zplib.image import colorize
from zplib.image import pyramid
from zplib.curve import spline_geometry
from zplib.curve import interpolate
import freeimage
import numpy
import pickle
import torch

from scipy.ndimage import gaussian_filter

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
    self.WIDTH_TRENDS = trend_data

self.AVG_WIDTHS = numpy.array([numpy.interp(5, self.WIDTH_TRENDS['ages'], wt) for wt in self.WIDTH_TRENDS['width_trends']])
AVG_WIDTHS_TCK = self.to_tck(self.AVG_WIDTHS)
self.AVG_WIDTHS_TCK = (AVG_WIDTHS_TCK[0], AVG_WIDTHS_TCK[1]/downscale, AVG_WIDTHS_TCK[2])
SCALE = [0,1,2,3]

