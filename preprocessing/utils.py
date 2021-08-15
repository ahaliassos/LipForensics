"""
Utils for pre-processing data. Adapted from https://github.com/mpc001/
Lipreading_using_Temporal_Convolutional_Networks/blob/master/preprocessing/transform.py"""

import numpy as np
from skimage import transform as tf


def warp_img(src, dst, img, std_size):
    """ "
    Warp image to match mean face landmarks

    Parameters
    ----------
    src : numpy.array
        Key Landmarks of initial face
    dst : numpy.array
        Key landmarks of mean face
    img : numpy.array
        Frame to be aligned
    std_size : tuple
        Target size for frames
    """
    tform = tf.estimate_transform("similarity", src, dst)  # find the transformation matrix
    warped = tf.warp(img, inverse_map=tform.inverse, output_shape=std_size)  # wrap the frame image
    warped = warped * 255
    warped = warped.astype("uint8")
    return warped, tform


def apply_transform(transform, img, std_size):
    """ "
    Apply affine transform to image.

    Parameters
    ----------
    transform : skimage.transform._geometric.GeometricTransform
        Object with transformation parameters
    img : numpy.array
        Frame to be aligned
    std_size : tuple
        Target size for frames
    """
    warped = tf.warp(img, inverse_map=transform.inverse, output_shape=std_size)
    warped = warped * 255
    warped = warped.astype("uint8")
    return warped


def cut_patch(img, landmarks, height, width, threshold=5):
    """ "
    Crop square mouth region given landmarks

    Parameters
    ----------
    img : numpy.array
        Frame to be cropped
    landmarks : numpy.array
        Landmarks corresponding to mouth region
    height : int
        Height of output image
    width : int
        Width of output image
    threshold : int, optional
        Threshold for determining whether to throw an exception when the initial bounding box is out of bounds

    """
    center_x, center_y = np.mean(landmarks, axis=0)
    if center_y - height < 0:
        center_y = height
    if int(center_y) - height < 0 - threshold:
        raise Exception("too much bias in height")
    if center_x - width < 0:
        center_x = width
    if center_x - width < 0 - threshold:
        raise Exception("too much bias in width")
    if center_y + height > img.shape[0]:
        center_y = img.shape[0] - height
    if center_y + height > img.shape[0] + threshold:
        raise Exception("too much bias in height")
    if center_x + width > img.shape[1]:
        center_x = img.shape[1] - width
    if center_x + width > img.shape[1] + threshold:
        raise Exception("too much bias in width")

    img_cropped = np.copy(
        img[
            int(round(center_y) - round(height)) : int(round(center_y) + round(height)),
            int(round(center_x) - round(width)) : int(round(center_x) + round(width)),
        ]
    )
    return img_cropped
