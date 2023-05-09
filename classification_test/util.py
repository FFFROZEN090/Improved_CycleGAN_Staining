"""This module contains simple helper functions """
import numpy as np
from PIL import Image
import os
import sys
import pathlib

def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)


# retrieve correct path depending on os and sds
def check_os(sds=False):
    path = ''
    if sys.platform == "linux":
        if sds:
            path = '/sds_hd/sd18a006/'
        path1 = '/home/marlen/'
        path2 = '/home/mr38/'
        if pathlib.Path('/home/marlen/').exists():
            return path1 + path
        elif pathlib.Path('/home/mr38/').exists():
            return path2 + path
        else:
            print('error: sds path cannot be defined! Abort')
            return 1
    elif sys.platform == "win32":
        path = ''
        if sds:
            path = '//lsdf02.urz.uni-heidelberg.de/sd18A006/'
        else:
            path = 'C:/Users/mr38/'
        if pathlib.Path(path).exists():
            return path
        else:
            print('error: sds path cannot be defined! Abort')
            return 1
    else:
        print('error: sds path cannot be defined! Abort')
        return 1

def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)
