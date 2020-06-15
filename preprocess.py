import cv2 
import numpy as np
import scipy as sc
from PIL import ImageOps, Image, ImageDraw

from skimage import segmentation
from skimage.draw import polygon2mask
from skimage.measure import find_contours
from skimage.filters import difference_of_gaussians

from matplotlib.pyplot import imshow
from pylab import *


def threshold(im, binary=False):
    '''
    im: array
    TODO:
    - noise filtering
    '''
    #im = histeq(im)
    #im = contrast(im)
    #imshow(im)
    
    minval, maxval = 0, 255
    
    if len(im.shape)  > 2:
        im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    
    im = cv2.normalize(im, None, alpha=minval, beta=maxval, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    
    if binary:
        minval = 5
        thresh_type = cv2.THRESH_BINARY
    else:
        thresh_type = cv2.THRESH_BINARY+cv2.THRESH_OTSU
        
    retval, im = cv2.threshold(im,minval,maxval,thresh_type)
    
    return im


def histeq(im,nbr_bins=256):
    """ Histogram equalization of a grayscale image. """
    '''
    Old version
    
    # get image histogram
    imhist,bins = histogram(im.flatten(),nbr_bins,normed=False)
    cdf = imhist.cumsum() # cumulative distribution function
    cdf = 255 * cdf / cdf[-1] # normalize
    # use linear interpolation of cdf to find new pixel values
    im2 = interp(im.flatten(),bins[:-1],cdf)
    im = im2.reshape(im.shape)
    '''
    if type(im) is not Image:
        im = Image.fromarray(im)
        
    im = ImageOps.equalize(im)
    
    if type(im) is not np.ndarray:
        im = np.array(im)
    
    return im


def contrast(im, cutoff=0, ignore=None):
    if type(im) is not Image:
        im = Image.fromarray(im)
        
    im = ImageOps.autocontrast(im, cutoff=cutoff, ignore=ignore)
    
    if type(im) is not np.ndarray:
        im = np.array(im)
    
    return im


def denoise(im,h=10,tw=7,sw=21):
    if type(im) is not np.ndarray:
        im = np.array(im)
        
    #im = cv2.GaussianBlur(im,(5,5),0)
    if is_colored(im):
        im = cv2.fastNlMeansDenoisingColored(im,None,h=h, hColor=h, templateWindowSize=tw, searchWindowSize=sw)
    else:
        im = cv2.fastNlMeansDenoising(im,None,h=h, templateWindowSize=tw, searchWindowSize=sw)
    
    return im


def get_foreground(im):
    im_processed = cv2.Laplacian(im,cv2.CV_8UC1)
    im2_processed = preprocess.denoise(im2_processed, h=3)
    #im_processed = cv2.dilate(im_processed,ones((3,3)),iterations=1)
    im_processed = threshold(im_processed, binary=False)
    
    h, w = im.shape[:2]
    seed = (int(h/2),int(w/2))
    #ImageDraw.floodfill(foreground, (h/2-2,w/2-8), 255, border=None, thresh=255)
    foreground = segmentation.flood_fill(im_processed, seed, 255, connectivity=0, tolerance=0, in_place=False)
    
    return foreground


def get_foreground2(im):
    im_thresh = threshold(im)
    contours = find_contours(im_thresh,.9,fully_connected='high', positive_orientation='low')
    msk = polygon2mask(im.shape[:2],contours[0]).astype('uint8')
    
    return msk


def get_rand_points(msk,size=5):
    non_black = np.argwhere(msk > 0)
    rand_points = np.random.randint(0, non_black.shape[0], size=size)
    
    return non_black[rand_points]


def is_colored(im):
    return len(im.shape) > 2


def recolor_bw(im, new_color):
    thresh = 30
    msk = (im < thresh)
    im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
    im[msk] = new_color
    
    return im


def enhance_edges(im):
    im_edge_enh = difference_of_gaussians(im, 1.5, multichannel=False)#, mode='constant', cval=1)
    im_edge_enh = cv2.normalize(im_edge_enh,None,alpha=0,beta=255, norm_type=cv2.NORM_MINMAX).astype('uint8')
    
    return im_edge_enh


def detect_edges(im, enhance=True):
    im_processed = im.copy()
    
    if enhance:
        im_processed = enhance_edges(im_processed)
        
    im_processed = cv2.Laplacian(im_processed,cv2.CV_8UC1)
    im_processed = cv2.dilate(im_processed,ones((5,5)),3)
    im_processed = threshold(im_processed, binary=False)
    
    return im_processed



def not_white(im):
    channels = np.ones((1,3))
    lower, upper = 0 * channels,200 * channels
    msk = cv2.inRange(im, lower, upper)
    return msk


def not_black(im):
    channels = np.ones((1,3))
    lower, upper = 100 * channels,255 * channels
    msk = cv2.inRange(im, lower, upper)
    return msk



def brush_paint(im):
    pass

def enhance_details(im):
    #im = cv2.detailEnhance(im,sigma_s=10, sigma_r=.15)
    pass
    