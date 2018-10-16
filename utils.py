from itertools import islice
import numpy as np
import random


class StaticRandomCrop(object):
    def __init__(self, image_size, crop_size):
        self.th, self.tw = crop_size
        h, w = image_size
        self.h1 = random.randint(0, h - self.th)
        self.w1 = random.randint(0, w - self.tw)

    def __call__(self, image):
        return image[self.h1:(self.h1+self.th), self.w1:(self.w1+self.tw)]


class StaticCenterCrop(object):
    def __init__(self, image_size, crop_size):
        self.th, self.tw = crop_size
        self.h, self.w = image_size
        
    def __call__(self, image):
        return image[(self.h-self.th)//2:(self.h+self.th)//2,
                     (self.w-self.tw)//2:(self.w+self.tw)//2]


def window(seq, n = 2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result

def one_hot(label, num_classes):
    label_onehot = np.zeros(num_classes, np.uint8)
    label_onehot[label] = 1
    return label_onehot

def get_size(origin_size = None, crop_size = None, resize_size = None, resize_scale = None):
    """ Get the resulting image size after cropping and resizing """
    if resize_size is not None:
        image_size = resize_size
    elif crop_size is not None:
        image_size = crop_size
    elif origin_size:
        image_size = origin_size
    else:
        raise ValueError('One of the argument should be not None')

    if resize_scale is not None:
        h, w = image_size[0]*resize_scale[0], image_size[1]*resize_scale[1]
        image_size = (h, w)

    return image_size
