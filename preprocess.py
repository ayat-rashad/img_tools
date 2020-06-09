import cv2 
import numpy as np
import scipy as sc
from PIL import ImageOps, Image
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


def denoise(im,h=3,tw=7,sw=21):
    if type(im) is not np.ndarray:
        im = np.array(im)
        
    #im = cv2.GaussianBlur(im,(5,5),0)
    im = cv2.fastNlMeansDenoising(im,None,h,tw,sw)
    
    return im


def recolor_bw(im, new_color):
    thresh = 30
    msk = (im < thresh)
    im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
    im[msk] = new_color
    
    return im


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
    