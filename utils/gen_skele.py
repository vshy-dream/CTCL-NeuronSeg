import cv2
from PIL import Image
import numpy as np
from skimage import morphology
# from stardist import edt_prob
from utils.edt_prob import edt_prob

def gen_skele_3d(labels):
    '''Extract 3d skeleton from 3d data
    '''
    ids = np.unique(labels)
    # print('the number of neurons: ', len(ids))
    skele_binary = np.zeros_like(labels, dtype=np.uint8)
    for id in ids:
        if id == 0:
            continue
        temp_lb = np.zeros_like(labels, dtype=np.uint8)
        temp_lb[labels == id] = 1
        # extract skeleton
        temp_skele = morphology.skeletonize_3d(temp_lb)
        # dilation
        # temp_skele = morphology.binary_dilation(temp_skele, selem=morphology.cube(3))
        temp_skele = morphology.binary_dilation(temp_skele, footprint=morphology.cube(3))
        skele_binary[temp_skele == 1] = 1

    # background is 0 and foreground is 0
    skele_binary = 1 - skele_binary
    return skele_binary

def gen_skele_dim(labels, dim='z'):
    '''Extract 2d skeleton on assigned dimenstion from 3d data
    Args:
        labels: numpy array, [Z, Y, X]
        dim: 'z', 'y' or 'x'
    '''
    if dim == 'y':
        labels = np.transpose(labels, (1, 0, 2))
    elif dim == 'x':
        labels = np.transpose(labels, (2, 0, 1))
    else:
        pass

    skele_binary = np.zeros_like(labels, dtype=np.uint8)
    num = labels.shape[0]
    for k in range(num):
        lb = labels[k]
        ids = np.unique(lb)
        for id in ids:
            if id == 0:
                continue
            temp_lb = np.zeros_like(lb, dtype=np.uint8)
            temp_lb[lb == id] = 1
            temp_skele = morphology.skeletonize(temp_lb)
            # temp_skele = morphology.binary_dilation(temp_skele, selem=morphology.square(3))
            temp_skele = morphology.binary_dilation(temp_skele, footprint=morphology.square(3))
            skele_binary[k][temp_skele == 1] = 1
    skele_binary = 1 - skele_binary

    if dim == 'y':
        skele_binary = np.transpose(skele_binary, (1, 0, 2))
    elif dim == 'x':
        skele_binary = np.transpose(skele_binary, (1, 2, 0))
    else:
        pass

    return skele_binary

def gen_distanceTransform(labels):
    prob = edt_prob(labels)
    return prob

