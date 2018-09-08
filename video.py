from torch.utils.data import Dataset
from pathlib import Path
from glob import glob
from itertools import islice, groupby
from fabric import colors
import numpy as np
import imageio
import torch
import random
random.seed(1)
import cv2
from functools import partial
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
    def __init__(self, dataset_dir, train_or_val,
                 strides = 3, stretchable = False,
                 cropper = 'random', crop_shape = None,
                 resize_shape = None, resize_scale = None):
        self.dataset_dir = dataset_dir
        self.train_or_val = train_or_val
        self.strides = strides
        self.stretchable = stretchable
        self.cropper = cropper
        self.crop_shape = crop_shape
        self.resize_shape = resize_shape
        self.resize_scale = resize_scale
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_paths = self.samples[idx]
        if self.stretchable:
            f_0, f_t, f_1 = sorted(np.random.choice(np.arange(self.strides), 3, replace = False))
            t = (f_t-f_0)/(f_1-f_0)
        else:
            f_0 = 0
            f_t = np.random.randint(1, self.strides-1)
            f_1 = self.strides-1
            t = f_t/f_1
            
        img0_path, imgt_path, img1_path = img_paths[f_0], img_paths[f_t], img_paths[f_1]
        img0, imgt, img1 = map(imageio.imread, (img0_path, imgt_path, img1_path))

        images = [img0, imgt, img1]
        if self.crop_shape is not None:
            cropper = StaticRandomCrop(img0.shape[:2], self.crop_shape) if self.cropper == 'random'\
              else StaticCenterCrop(img0.shape[:2], self.crop_shape)
            images = list(map(cropper, images))

        if self.resize_shape is not None:
            resizer = partial(cv2.resize, dsize = tuple(self.resize_shape[::-1]))
            images = list(map(resizer, images))

        elif self.resize_scale is not None:
            resizer = partial(cv2.resize, dsize = (0,0), fx = self.resize_scale, fy = self.resize_scale)
            images = list(map(resizer, images))

        return np.array(images), t

    def has_txt(self): 
        p = Path(self.dataset_dir) / (self.train_or_val + f'_{self.strides}frames.txt')
        self.samples = []
        with open(p, 'r') as f:
            for i in f.readlines():
                self.samples.append(i.strip().split(','))

    @abstractmethod
    def has_no_txt(self): ...
    
    def split(self, samples): # used when train/val set are not stated
        p = Path(self.dataset_dir)
        test_ratio = 0.1
        random.shuffle(samples)
        idx = int(len(samples) * (1 - test_ratio))
        train_samples = samples[:idx]
        val_samples = samples[idx:]

        with open(p / f'train_{self.strides}frames.txt', 'w') as f: f.writelines((','.join(i) + '\n' for i in train_samples))
        with open(p / f'val_{self.strides}frames.txt', 'w') as f: f.writelines((','.join(i) + '\n' for i in val_samples))

        self.samples = train_samples if self.train_or_val == 'train' else val_samples


# DAVIS
# ------------------------------------------------------
class DAVIS(BaseDataset):
    def __init__(self, dataset_dir, train_or_val, resolution = '480p',
                 strides = 3, stretchable = False,
                 cropper = 'random', crop_shape = None, resize_shape = None, resize_scale = None):
        # super(DAVIS, self).__init__()
        super().__init__(dataset_dir, train_or_val, strides, stretchable,
                         cropper, crop_shape, resize_shape, resize_scale)
        assert resolution in ['480p', 'Full-Resolution']
        self.resolution = resolution

        p = Path(dataset_dir) / (train_or_val + f'_{resolution}_{strides}frames.txt')
        if p.exists(): self.has_txt()
        else: self.has_no_txt()

    def has_txt(self): 
        p = Path(self.dataset_dir) / (self.train_or_val + f'_{self.resolution}_{self.strides}frames.txt')
        self.samples = []
        with open(p, 'r') as f:
            for i in f.readlines():
                self.samples.append(i.strip().split(','))
            
    def has_no_txt(self):
        p = Path(self.dataset_dir)
        p_set = p / 'ImageSets/2017' / (self.train_or_val+'.txt')
        p_img = p / 'JPEGImages' / self.resolution
        self.samples = []
        
        with open(p_set, 'r') as f:
            for i in f.readlines():
                p_img_categ = p_img / i.strip()
                collection = sorted(map(str, p_img_categ.glob('*.jpg')))
                self.samples += [i for i in window(collection, self.strides)]

        assert self.train_or_val in ['train', 'val'],\
          f'property train_or_val must be train/val (given {self.train_or_val})'

        with open(p / (self.train_or_val + f'_{self.resolution}_{self.strides}frames.txt'), 'w') as f:
            f.writelines((','.join(i) + '\n' for i in self.samples))
        print(colors.green('** DAVIS dataset originally has train/val.txt in DAVIS/ImageSets/2017,'+\
                           ' never confused by generated files (contents are same).**'))
    

# Sintel
# ============================================================
class Sintel(BaseDataset):
    def __init__(self, dataset_dir, train_or_val, mode = 'final',
                 strides = 3, stretchable = False,
                 cropper = 'random', crop_shape = None, resize_shape = None, resize_scale = None):
        super().__init__(dataset_dir, train_or_val, strides, stretchable,
                         cropper, crop_shape, resize_shape, resize_scale)
        self.mode = mode

        p = Path(dataset_dir) / (train_or_val + f'_{strides}frames.txt')
        if p.exists(): self.has_txt()
        else: self.has_no_txt()
    
    def has_no_txt(self):
        p = Path(self.dataset_dir)
        p_img = p / 'training' / self.mode
        samples = []

        collections_of_scenes = sorted(map(str, p_img.glob('**/*.png')))
        collections = [list(g) for k, g in groupby(collections_of_scenes, lambda x: x.split('/')[-2])]

        samples = [i for collection in collections for i in window(collection, self.strides)]
        self.split(samples)

class SintelFinal(Sintel):
    def __init__(self, dataset_dir, train_or_val, strides = 3, stretchable = False,
                 cropper = 'random', crop_shape = None, resize_shape = None, resize_scale = None):
        super(SintelFinal, self).__init__(dataset_dir, train_or_val, 'final', strides, stretchable,
                                          cropper, crop_shape, resize_shape, resize_scale)

class SintelClean(Sintel):
    def __init__(self, dataset_dir, train_or_val, strides = 3, stretchable = False,
                 cropper = 'random', crop_shape = None, resize_shape = None, resize_scale = None):
        super(SintelClean, self).__init__(dataset_dir, train_or_val, 'clean', strides, stretchable,
                                          cropper, crop_shape, resize_shape, resize_scale)

# KITTI
# ============================================================
# class KITTI(BaseDataset):

#     def __init__(self, dataset_dir, train_or_test, ):
#         pass

#     def has_no_txt(self):
#         pass


# UCF101 ! still in progress
# ------------------------------------
# class UCF101(BaseDataset):
#     def __init__(self, dataset_dir, train_or_val, color = 'rgb',
#                  cropper = 'random', crop_shape = None, resize_shape = None, resize_scale = None):
#         # super(DAVIS, self).__init__()
#         super().__init__()
#         self.color = color
#         self.cropper = cropper
#         self.crop_shape = crop_shape
#         self.resize_shape = resize_shape
#         self.resize_scale = resize_scale

#         self.dataset_dir = dataset_dir
#         self.train_or_val = train_or_val
#         p = Path(dataset_dir) / (train_or_val + '_triplet.txt')
#         if p.exists(): self.has_txt()
#         else: self.has_no_txt()

#     def __getitem__(self, idx):
#         avi_path = self.samples[idx]
        
#         img1, img2, img3 = map(imageio.imread, (img1_path, img2_path, img3_path))

#         if self.color == 'gray':
#             img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)[:,:,np.newaxis]
#             img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)[:,:,np.newaxis]
#             img3 = cv2.cvtColor(img3, cv2.COLOR_RGB2GRAY)[:,:,np.newaxis]

#         images = [img1, img2, img3]
#         if self.crop_shape is not None:
#             cropper = StaticRandomCrop(img1.shape[:2], self.crop_shape) if self.cropper == 'random'\
#               else StaticCenterCrop(img1.shape[:2], self.crop_shape)
#             images = list(map(cropper, images))

#         if self.resize_shape is not None:
#             resizer = partial(cv2.resize, dsize = (0,0), dst = self.resize_shape)
#             images = list(map(resizer, images))

#         elif self.resize_scale is not None:
#             resizer = partial(cv2.resize, dsize = (0,0), fx = self.resize_scale, fy = self.resize_scale)
#             images = list(map(resizer, images))

#         return np.array(images)

#     def has_txt(self): 
#         p = Path(self.dataset_dir) / (self.train_or_val + '.txt')
#         self.samples = []
#         with open(p, 'r') as f:
#             for i in f.readlines():
#                 img1, img2, img3 = i.split(',')
#                 self.samples.append((img1, img2, img3))

#     @abstractmethod
#     def has_no_txt(self): ...
    
#     def split(self, samples): # used when train/val set are not stated
#         p = Path(self.dataset_dir)
#         test_ratio = 0.1
#         random.shuffle(samples)
#         idx = int(len(samples) * (1 - test_ratio))
#         train_samples = samples[:idx]
#         val_samples = samples[idx:]

#         with open(p / 'train_triplet.txt', 'w') as f: f.writelines((','.join(i) + '\n' for i in train_samples))
#         with open(p / 'val_triplet.txt', 'w') as f: f.writelines((','.join(i) + '\n' for i in val_samples))

#         self.samples = train_samples if self.train_or_val == 'train' else val_samples
    

class NeedforSpeed(BaseDataset):
    def __init__(self, dataset_dir, train_or_val, color = 'rgb', strides = 8, stretchable = False,
                 cropper = 'random', crop_shape = (384, 448), resize_shape = None, resize_scale = None):
        super().__init__()
        self.color = color
        self.strides = strides
        self.stretchable = stretchable
        self.cropper = cropper
        self.crop_shape = crop_shape
        self.resize_shape = resize_shape
        self.resize_scale = resize_scale

        self.dataset_dir = dataset_dir
        self.train_or_val = train_or_val
        p = Path(dataset_dir) / (train_or_val + f'_{strides}frames.txt')
        if p.exists(): self.has_txt()
        else: self.has_no_txt()

    def has_no_txt(self):
        p = Path(self.dataset_dir)
        samples = []

        print('Collecting NeedforSpeed image paths, this operation can take a few minutes ...')
        collections_of_scenes = sorted(map(str, p.glob('**/240/*/*.jpg')))
        collections = [list(g) for k, g in groupby(collections_of_scenes, lambda x: x.split('/')[-2])]
        
        samples = [i for collection in collections for i in window(collection, self.strides)]
        self.split(samples)
        
# def get_dataset(dataset_name):
#     return {
#         'DAVIS': DAVIS,
#         'Sintel': Sintel,
#         'SintelClean': SintelClean,
#         'SintelFinal': SintelFinal,
#         # 'FlyingChairs': FlyingChairs,
#     }[dataset_name]


