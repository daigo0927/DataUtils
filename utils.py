from itertools import islice
import numpy as np
import random

from . import image, video, semseg, flow


class StaticRandomCrop(object):
    def __init__(self, image_size, crop_size):
        self.th, self.tw = crop_size
        h, w = image_size
        self.h1 = random.randint(0, h - self.th)
        self.w1 = random.randint(0, w - self.tw)

    def __call__(self, img):
        return img[self.h1:(self.h1+self.th), self.w1:(self.w1+self.tw)]


class StaticCenterCrop(object):
    def __init__(self, image_size, crop_size):
        self.th, self.tw = crop_size
        self.h, self.w = image_size
        
    def __call__(self, img):
        return img[(self.h-self.th)//2:(self.h+self.th)//2, (self.w-self.tw)//2:(self.w+self.tw)//2]


def window(seq, n=2):
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

def txtread(uri): # to read label file encoded by .txt (e.g. SYNTHIA dataset)
    with open(uri, 'r') as f:
        label = np.array([list(map(int, i.strip().split())) for i in f.readlines()])
    return label

def get_dataset(dataset_name):
    return {
        'Food-101':image.Food101,
        'DAVIS_v': video.DAVIS,
        'Sintel_v': video.Sintel,
        'SintelClean_v': video.SintelClean,
        'SintelFinal_v': video.SintelFinal,
        # 'UCF101': UCF101,
        'NeedforSpeed': video.NeedforSpeed, 'NeedforSpeed_v': video.NeedforSpeed,
        'CityScapes': semseg.CityScapes, 'CityScapes_s': semseg.CityScapes,
        'SYNTHIA': semseg.SYNTHIA, 'SYNTHIA_s': semseg.SYNTHIA,
        'PlayingforData': semseg.PlayingforData, 'PlayingforData_s': semseg.PlayingforData,
        'FlyingChairs': flow.FlyingChairs, 'FlyingChairs_f': flow.FlyingChairs,
        # 'FlyingThings': flow.FlyingThings,
        'Sintel': flow.Sintel, 'Sintel_f': flow.Sintel,
        'SintelClean': flow.SintelClean, 'SintelClean_f': flow.SintelClean,
        'SintelFinal': flow.SintelFinal, 'SintelFinal_f': flow.SintelFinal,
        'KITTI': flow.KITTI, 'KITTI_f': flow.KITTI
    }[dataset_name]
