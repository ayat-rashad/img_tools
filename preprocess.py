import cv2 
import numpy as np
import scipy as sc
from PIL import ImageOps, Image, ImageDraw

from skimage import segmentation, exposure
from skimage.draw import polygon2mask
from skimage.measure import find_contours
from skimage.filters import difference_of_gaussians
from skimage.filters import rank
from skimage import filters
from skimage import morphology
from skimage import exposure
from skimage.exposure import match_histograms

from matplotlib import pyplot as plt
from matplotlib.pyplot import imshow
#from pylab import *


def threshold(im, binary=False, local=False):
    '''
    im: array
    TODO:
    - noise filtering
    - use local otsu: rank.otsu
    '''
    #im = histeq(im)
    #im = contrast(im)
    #imshow(im)
    
    minval, maxval = 0, 255
    
    im_gray = im.copy()
    
    if is_colored(im):
        im_gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    
    #im_gray = cv2.normalize(im_gray, None, alpha=minval, beta=maxval, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    
    if binary:
        minval = 5
        thresh_type = cv2.THRESH_BINARY
        retval, im_thresh = cv2.threshold(im_gray,minval,maxval,thresh_type)
    elif local == False:
        #thresh_type = cv2.THRESH_BINARY+cv2.THRESH_OTSU
        thresh = filters.threshold_otsu(im_gray)
        im_thresh = (im_gray > thresh).astype('uint8')*255
    else:
        im_thresh = rank.otsu(im_gray, morphology.disk(5))
        im_thresh = (im_gray >= im_thresh).astype('uint8')*255
        
    return im_thresh


def threshold_local(im, thresh_type='sauvola'):    
    minval, maxval = 0, 255
    
    im2 = im.copy()
    
    if is_colored(im):
        im2 = cv2.cvtColor(im2, cv2.COLOR_RGB2GRAY)
    
    im2 = cv2.normalize(im2, None, alpha=minval, beta=maxval, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    
    if thresh_type == 'sauvola':    
        im2_thresh = filters.threshold_sauvola(im2, window_size=25)
        
    im2 = (im2 > im2_thresh).astype('uint8')*255
    
    return im2


def histeq(im,nbr_bins=256):
    '''
    TODO:
    - use local method.
    
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
    im_processed = denoise(im_processed, h=3)
    #im_processed = cv2.dilate(im_processed,ones((3,3)),iterations=1)
    im_processed = threshold_local(im_processed)
    
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


def get_foreground_slic(im):
    '''
    TODO:
    - make sure the foreground object is white
    '''
    im_slic = segmentation.slic(im,3, start_label=1, compactness=20.0, enforce_connectivity=False, max_iter=100)
    im_slic = segmentation.clear_border(im_slic)
    #im_slic = threshold(im_slic)
    #im_slic = threshold_local(im_slic)
    im_slic = (im_slic > 0).astype('uint8') * 255
    
    return im_slic


def merge_segmentations(s1, s2):
    merged_s = segmentation.join_segmentations(s1, s2)
    
    return merged_s


def get_rand_points(msk,size=5):
    '''
    TODO:
    - try skimage.util.regular_seeds
    '''
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
    im_edge_enh = difference_of_gaussians(im, 1.5, multichannel=False) * 100#, mode='constant', cval=1)
    #im_edge_enh = cv2.normalize(im_edge_enh,None,alpha=0,beta=255, norm_type=cv2.NORM_MINMAX).astype('uint8')
    im_edge_enh = exposure.rescale_intensity(im_edge_enh.astype('int8'), out_range=(0,255)).astype('uint8')
    
    return im_edge_enh


def detect_edges(im, enhance=False):
    im_processed = im.copy()
    
    if enhance:
        im_processed = enhance_edges(im_processed)
        
    im_processed = cv2.Laplacian(im_processed,cv2.CV_8UC1)
    im_processed = cv2.dilate(im_processed,np.ones((5,5)),3)
    im_processed = threshold(im_processed, binary=False)
    
    return im_processed


def hist_mtch_cs(im1, im2, color_space='RGB', transfer_channels=[0,1,2]):
    trans_1 = trans_2 = None
        
    if color_space == 'LAB':
        trans_1 = cv2.COLOR_RGB2LAB; trans_2 = cv2.COLOR_LAB2RGB
        
    if color_space == 'HSV':
        trans_1 = cv2.COLOR_RGB2HSV; trans_2 = cv2.COLOR_HSV2RGB
        
    if color_space == 'HLS':
        trans_1 = cv2.COLOR_RGB2HLS; trans_2 = cv2.COLOR_HLS2RGB
        
    im1_2 = im1.copy()
    im2_2 = im2.copy()
    
    if trans_1 is not None:
        im1_2 = cv2.cvtColor(im1, trans_1)
        im2_2 = cv2.cvtColor(im2, trans_1)
        
    matched = match_histograms(im1_2, im2_2, multichannel=True)
    
    for c in transfer_channels:
        im1_2[:,:,c] = matched[:,:,c]
        
    if trans_2 is not None:
        im1_2 = cv2.cvtColor(im1_2, trans_2)
        
    return im1_2



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


def remove_background(im, foreground_msk, bg_val='white', bbox=None):    
    if bbox is not None:
        min_r, min_c, max_r, max_c = bbox
        raw_bg = np.ones((max_r-min_r,max_c-min_c, 3))
        msk = foreground_msk[min_r:max_r, min_c:max_c]
    else:
        raw_bg = np.ones_like(im)
        msk = foreground_msk.copy()
        
    if bg_val == 'white':
        bg = raw_bg * 255
    elif bg_val == 'mean':
        bg = (im.mean(axis=(0,1)) * raw_bg).astype('uint8')
    elif bg_val == 'median':
        bg = (np.percentile(im,50,axis=(0,1)) * raw_bg).astype('uint8')
    elif bg_val == 'min':
        bg = (np.percentile(im,20,axis=(0,1)) * raw_bg).astype('uint8')
    elif bg_val == 'max':
        bg = (np.percentile(im,80,axis=(0,1)) * raw_bg).astype('uint8')
    else:
        bg = raw_bg * bg_val
        
    bg[msk>0] = im[foreground_msk>0]
    
    return bg


def reload_background(im, im_ref, foreground_msk, bbox=None):
    im_final = im_ref.copy()
    msk = foreground_msk.copy()
    
    if bbox is not None:
        min_r, min_c, max_r, max_c = bbox
        msk = foreground_msk[min_r:max_r, min_c:max_c]
        
    im_final[foreground_msk>0] = im[msk>0]
    
    return im_final


def hist_eq_cs(im, color_space='LAB', eq_channels=[0], k_size=150, use_plot=False, clip=0.01):
    trans_1 = trans_2 = None
        
    if color_space == 'LAB':
        trans_1 = cv2.COLOR_RGB2LAB; trans_2 = cv2.COLOR_LAB2RGB; light_ch = 0
        
    if color_space == 'HSV':
        trans_1 = cv2.COLOR_RGB2HSV; trans_2 = cv2.COLOR_HSV2RGB; light_ch = 2
        
    if color_space == 'HLS':
        trans_1 = cv2.COLOR_RGB2HLS; trans_2 = cv2.COLOR_HLS2RGB; light_ch = 1
        
    if color_space == 'LUV':
        trans_1 = cv2.COLOR_RGB2LUV; trans_2 = cv2.COLOR_LUV2RGB; light_ch = 0
        
    im1 = im.copy()
    
    if trans_1 is not None:
        im1 = cv2.cvtColor(im1, trans_1)
            
    for c in eq_channels:
        im1[:,:,c] = (exposure.equalize_adapthist(im1[:,:,c],k_size, clip_limit=clip) * 255).astype('uint8')
        
    if use_plot:
        imshow_c(im[:,:,light_ch])
        
        hist, h_centers = exposure.histogram(im[:,:,light_ch])
        plot2(h_centers,hist)
        
        imshow_c(im1[:,:,light_ch])
        
        hist, h_centers = exposure.histogram(im1[:,:,light_ch])
        plot2(h_centers,hist)
        
        
    if trans_2 is not None:
        im1 = cv2.cvtColor(im1, trans_2)
        
    return im1
    

def imshow_g(im):
    plt.figure()
    plt.imshow(im,cm.gray,vmin=0,vmax=255)
    
    
def imshow_c(im):
    plt.figure()
    plt.imshow(im)
    
    
def plot2(x,y):
    plt.figure()
    plt.plot(x,y)


def crop_bbox(im, bbox):
    min_r, min_c, max_r, max_c = bbox
    new_im = np.ones((max_r-min_r,max_c-min_c, 3))
    new_im = im[min_r:max_r, min_c:max_c]
    
    return new_im


def crop_main_bbox(im, bboxs):
    max_area = 0
    main_bbox = None
    
    for bbox in bboxs:
        min_r, min_c, max_r, max_c = bbox
        area = (max_r - min_r) * (max_c - min_c)
        
        if area > max_area:
            max_area = area
            main_bbox = bbox
            
    min_r, min_c, max_r, max_c = bbox
    new_im = im[min_r:max_r, min_c:max_c]
    
    return new_im, main_bbox