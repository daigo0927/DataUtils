from torch.utils.data import Dataset
from pathlib import Path
from itertools import islice
import numpy as np
import imageio
import torch
import random
import cv2
from functools import partial
# from flow_utils import load_flow
from abc import abstractmethod, ABCMeta


class StaticRandomCrop(object):
    def __init__(self, image_size, crop_size):
        self.th, self.tw = crop_size
        h, w = image_size
        self.h1 = random.randint(0, h - self.th)
        self.w1 = random.randint(0, w - self.tw)

    def __call__(self, img):
        return img[self.h1:(self.h1+self.th), self.w1:(self.w1+self.tw),:]


class StaticCenterCrop(object):
    def __init__(self, image_size, crop_size):
        self.th, self.tw = crop_size
        self.h, self.w = image_size
    def __call__(self, img):
        return img[(self.h-self.th)//2:(self.h+self.th)//2, (self.w-self.tw)//2:(self.w+self.tw)//2,:]


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


class BaseDataset(Dataset, metaclass = ABCMeta):
    @abstractmethod
    def __init__(self): pass
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        img1_path, img2_path, img3_path = self.samples[idx]
        img1, img2, img3 = map(imageio.imread, (img1_path, img2_path, img3_path))

        if self.color == 'gray':
            img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)[:,:,np.newaxis]
            img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)[:,:,np.newaxis]
            img3 = cv2.cvtColor(img3, cv2.COLOR_RGB2GRAY)[:,:,np.newaxis]

        images = [img1, img2, img3]
        if self.crop_shape is not None:
            cropper = StaticRandomCrop(img1.shape[:2], self.crop_shape) if self.cropper == 'random' else StaticCenterCrop(img1.shape[:2], self.crop_shape)
            images = list(map(cropper, images))

        if self.resize_shape is not None:
            resizer = partial(cv2.resize, dsize = (0,0), dst = self.resize_shape)
            images = list(map(resizer, images))

        elif self.resize_scale is not None:
            resizer = partial(cv2.resize, dsize = (0,0), fx = self.resize_scale, fy = self.resize_scale)
            images = list(map(resizer, images))

        return np.array(images)

    @abstractmethod
    def has_txt(self): ...
        # p = Path(self.dataset_dir) / (self.train_or_test + '.txt')
        # self.samples = []
        # with open(p, 'r') as f:
        #     for i in f.readlines():
        #         img1, img2, flow = i.split(',')
        #         flow = flow.strip()
        #         self.samples.append((img1, img2, flow))

    @abstractmethod
    def has_no_txt(self): ...
    
    def split(self, samples):
        p = Path(self.dataset_dir)
        test_ratio = 0.1
        random.shuffle(samples)
        idx = int(len(samples) * (1 - test_ratio))
        train_samples = samples[:idx]
        test_samples = samples[idx:]

        with open(p / 'train.txt', 'w') as f: f.writelines((','.join(i) + '\n' for i in train_samples))
        with open(p / 'test.txt', 'w') as f: f.writelines((','.join(i) + '\n' for i in test_samples))

        self.samples = train_samples if self.train_or_test == 'train' else test_samples


# DAVIS
# ------------------------------------------------------
class DAVIS(BaseDataset):

    def __init__(self, dataset_dir, train_or_val, color = 'rgb', resolution = '480p',
                 cropper = 'random', crop_shape = None, resize_shape = None, resize_scale = None):
        # super(DAVIS, self).__init__()
        super().__init__()
        self.resolution = resolution
        self.color = color
        self.cropper = cropper
        self.crop_shape = crop_shape
        self.resize_shape = resize_shape
        self.resize_scale = resize_scale

        self.dataset_dir = dataset_dir
        self.train_or_val = train_or_val
        p_set = Path(dataset_dir) / 'ImageSets/2017' / (train_or_val+'.txt')
        p_img = Path(dataset_dir) / 'JPEGImages' / self.resolution
        self.samples = []
        
        with open(p_set, 'r') as f:
            for i in f.readlines():
                p_img_categ = p_img / i.strip()
                p_img_categ_imgs = sorted(map(str, p_img_categ.glob('*.jpg')))
                for j in range(len(p_img_categ_imgs)-2):
                    self.samples.append((p_img_categ_imgs[j],
                                         p_img_categ_imgs[j+1],
                                         p_img_categ_imgs[j+2]))
    def has_txt(self):
        pass

    def has_no_txt(self):
        pass
    

# Sintel
# ============================================================
class Sintel(BaseDataset):

    def __init__(self, dataset_dir, train_or_test, mode = 'final', color = 'rgb', cropper = 'random', crop_shape = None, resize_shape = None, resize_scale = None):
        super(Sintel, self).__init__()
        self.mode = mode
        self.color = color
        self.cropper = cropper
        self.crop_shape = crop_shape
        self.resize_shape = resize_shape
        self.resize_scale = resize_scale

        self.dataset_dir = dataset_dir
        self.train_or_test = train_or_test
        p = Path(dataset_dir) / (train_or_test + '.txt')
        if p.exists(): self.has_txt()
        else: self.has_no_txt()
    
    def has_no_txt(self):
        p = Path(self.dataset_dir)
        p_img = p / 'training' / self.mode
        p_flow = p / 'training/flow'
        samples = []

        collections_of_scenes = sorted(map(str, p_img.glob('**/*.png')))
        from itertools import groupby
        collections = [list(g) for k, g in groupby(collections_of_scenes, lambda x: x.split('/')[-2])]

        samples = [(*i, i[0].replace(self.mode, 'flow').replace('.png', '.flo')) for collection in collections for i in window(collection, 2)]
        self.split(samples)

class SintelFinal(Sintel):
    def __init__(self, dataset_dir, train_or_test, color = 'rgb', cropper = 'random', crop_shape = None, resize_shape = None, resize_scale = None):
        super(SintelFinal, self).__init__(dataset_dir, train_or_test, mode = 'final', color = color, cropper = cropper, crop_shape = crop_shape, resize_shape = resize_shape, resize_scale = resize_scale)

class SintelClean(Sintel):
    def __init__(self, dataset_dir, train_or_test, color = 'rgb', cropper = 'random', crop_shape = None, resize_shape = None, resize_scale = None):
        super(SintelClean, self).__init__(dataset_dir, train_or_test, mode = 'clean', color = color, cropper = cropper, crop_shape = crop_shape, resize_shape = resize_shape, resize_scale = resize_scale)

# KITTI
# ============================================================
class KITTI(BaseDataset):

    def __init__(self, dataset_dir, train_or_test, ):
        pass

    def has_no_txt(self):
        pass

def get_dataset(dataset_name):
    return {
        'Sintel': Sintel,
        'SintelClean': SintelClean,
        'SintelFinal': SintelFinal,
        'FlyingChairs': FlyingChairs,
    }[dataset_name]


