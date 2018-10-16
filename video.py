import numpy as np
import random
import cv2
from pathlib import Path
from imageio import imread
from itertools import groupby
from functools import partial
from fabric import colors
from torch.utils.data import Dataset
from abc import abstractmethod, ABCMeta

from . import utils


class BaseDataset(Dataset, metaclass = ABCMeta):
    """ Abstract class to flexibly utilize torch.util.data.Dataset """
    def __init__(self, dataset_dir, train_or_val,
                 strides = 3, stretchable = False, origin_size = None,
                 crop_type = 'random', crop_shape = None,
                 resize_shape = None, resize_scale = None):
        """ 
        Args:
        - dataset_dir str: target dataset directory
        - train_or_val str: flag indicates train or validation
        - strides int: target temporal range of triplet images
        - stretchable bool: enabling shift of start and end index of triplet images
        - origin_size tuple<int>: original size of target images
        - crop_type str: crop type either of [random, center, None]
        - crop_shape tuple<int>: crop shape
        - resize_shape tuple<int>: resize shape
        - resize_scale tuple<int>: resize scale
        """
        self.dataset_dir = dataset_dir
        self.train_or_val = train_or_val
        
        self.strides = strides
        self.stretchable = stretchable

        self.image_size = utils.get_size(origin_size, crop_shape, resize_shape, resize_scale)
        self.crop_type = crop_type
        self.crop_shape = crop_shape
        self.resize_shape = resize_shape
        self.resize_scale = resize_scale

        p = Path(dataset_dir) / (train_or_val + f'_{strides}frames.txt')
        if p.exists(): self.has_txt()
        else: self.has_no_txt()
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        paths = self.samples[idx]
        if self.stretchable:
            f, f_end = sorted(np.random.choice(range(1, self.strides), 2, replace = False))
        else:
            f_end = self.strides-1
            f = np.random.randint(1, f_end)

        t = np.array(f/f_end)
            
        image_0_path, image_t_path, image_1_path = paths[0], paths[f], paths[f_end]
        image_0, image_t, image_1 = map(imread, [image_0_path, image_t_path, image_1_path])

        if self.crop_shape is not None:
            cropper = utils.StaticRandomCrop(image_0.shape[:2], self.crop_shape) if self.crop_type == 'random'\
              else utils.StaticCenterCrop(image_0.shape[:2], self.crop_shape)
            image_0, image_t, image_1 = map(cropper, [image_0, image_t, image_1])

        if self.resize_shape is not None:
            resizer = partial(cv2.resize, dsize = tuple(self.resize_shape[::-1]))
            image_0, image_t, image_1 = map(resizer, [image_0, image_t, image_1])

        elif self.resize_scale is not None:
            sx, sy = self.resize_scale
            resizer = partial(cv2.resize, dsize = (0,0), fx = sx, fy = sy)
            image_0, image_t, image_1 = map(resizer, [image_0, image_t, image_1])

        images = np.stack([image_0, image_t, image_1], axis = 0)
        return images, t

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
        val_ratio = 0.1
        random.shuffle(samples)
        idx = int(len(samples) * (1 - val_ratio))
        train_samples = samples[:idx]
        val_samples = samples[idx:]

        with open(p / f'train_{self.strides}frames.txt', 'w') as f: f.writelines((','.join(i) + '\n' for i in train_samples))
        with open(p / f'val_{self.strides}frames.txt', 'w') as f: f.writelines((','.join(i) + '\n' for i in val_samples))

        self.samples = train_samples if self.train_or_val == 'train' else val_samples


# DAVIS
# ------------------------------------------------------
class DAVIS(BaseDataset):
    """ DAVIS dataset pipeline """
    def __init__(self, dataset_dir, train_or_val, resolution = '480p',
                 strides = 3, stretchable = False, origin_size = None,
                 crop_type = 'random', crop_shape = None,
                 resize_shape = None, resize_scale = None):
        """ 
        Args:
        - dataset_dir str: target dataset directory
        - train_or_val str: flag indicates train or validation
        - resolution str: either 480p or Full-Resolution for target resolution
        - strides int: target temporal range of triplet images
        - stretchable bool: enabling shift of start and end index of triplet images
        - origin_size tuple<int>: original size of target images
        - crop_type str: crop type either of [random, center, None]
        - crop_shape tuple<int>: crop shape of target images
        - resize_shape tuple<int>: resize shape
        - resize_scale tuple<int>: resize scale
        """
        if not resolution in ['480p', 'Full-Resolution']:
            raise ValueError('Invalid argument for target resolution')
        self.resolution = resolution
        super().__init__(dataset_dir, train_or_val, strides, stretchable,
                         origin_size, crop_type, crop_shape,
                         resize_shape, resize_scale)

    def has_txt(self): 
        p = Path(self.dataset_dir) / (self.train_or_val + f'_{self.strides}frames.txt')
        res_other = 'Full-Resolution' if self.resolution == '480p' else '480p'
        self.samples = []
        with open(p, 'r') as f:
            for i in f.readlines():
                self.samples.append(i.replace(res_other, self.resolution).strip().split(','))
            
    def has_no_txt(self):
        p = Path(self.dataset_dir)
        p_set = p / 'ImageSets/2017' / (self.train_or_val+'.txt')
        p_img = p / 'JPEGImages' / self.resolution
        
        self.samples = []
        with open(p_set, 'r') as f:
            for i in f.readlines():
                p_img_categ = p_img / i.strip()
                collection = sorted(map(str, p_img_categ.glob('*.jpg')))
                self.samples += [i for i in utils.window(collection, self.strides)]

        with open(p / (self.train_or_val+f'_{self.strides}frames.txt'), 'w') as f:
            f.writelines((','.join(i) + '\n' for i in self.samples))
        print(colors.green('** DAVIS dataset originally has train/val.txt in DAVIS/ImageSets/2017,'+\
                           ' never confused by generated files (contents are same).**'))
    

# Sintel
# ============================================================
class Sintel(BaseDataset):
    """ MPI-Sintel-complete dataset pipeline """
    def __init__(self, dataset_dir, train_or_val, mode = 'clean',
                 strides = 3, stretchable = False, origin_size = None,
                 crop_type = 'random', crop_shape = None,
                 resize_shape = None, resize_scale = None):
        """ 
        Args:
        - dataset_dir str: target dataset directory
        - train_or_val str: flag indicates train or validation
        - mode str: either clean or final to specify data path
        - strides int: target temporal range of triplet images
        - stretchable bool: enabling shift of start and end index of triplet images
        - origin_size tuple<int>: original size of target images
        - crop_type str: crop type either of [random, center, None]
        - crop_shape tuple<int>: crop shape of target images
        - resize_shape tuple<int>: resize shape
        - resize_scale tuple<int>: resize scale
        """
        self.mode = mode
        super().__init__(dataset_dir, train_or_val, strides, stretchable,
                         origin_size, crop_type, crop_shape,
                         resize_shape, resize_scale)
    
    def has_no_txt(self):
        p = Path(self.dataset_dir)
        p_img = p / 'training' / self.mode
        samples = []

        collections_of_scenes = sorted(map(str, p_img.glob('**/*.png')))
        collections = [list(g) for k, g in groupby(collections_of_scenes, lambda x: x.split('/')[-2])]
        samples = [(*i, i[0].replace(self.mode, 'flow').replace('.png', '.flo'))\
                    for collection in collections for i in utils.window(collection, 2)]
        self.split(samples)


class SintelClean(Sintel):
    """ MPI-Sintel-complete dataset (clean path) pipeline """
    def __init__(self, dataset_dir, train_or_val,
                 strides = 3, stretchable = False, origin_size = None,
                 crop_type = 'random', crop_shape = None,
                 resize_shape = None, resize_scale = None):
        super().__init__(dataset_dir, train_or_val, 'clean',
                         strides, stretchable, origin_size,
                         crop_type, crop_shape,
                         resize_shape, resize_scale)

    def has_txt(self):
        p = Path(self.dataset_dir) / (self.train_or_val+f'_{self.strides}.txt')
        self.samples = []
        with open(p, 'r') as f:
            for i in f.readlines():
                self.samples.append(i.replace('final', 'clean').strip().split(','))

        
class SintelFinal(Sintel):
    """ MPI-Sintel-complete dataset (clean path) pipeline """
    def __init__(self, dataset_dir, train_or_val,
                 strides = 3, stretchable = False, origin_size = None,
                 crop_type = 'random', crop_shape = None,
                 resize_shape = None, resize_scale = None):
        super().__init__(dataset_dir, train_or_val, 'final',
                         strides, stretchable, origin_size,
                         crop_type, crop_shape,
                         resize_shape, resize_scale)

    def has_txt(self):
        p = Path(self.dataset_dir) / (self.train_or_val+f'_{self.strides}.txt')
        self.samples = []
        with open(p, 'r') as f:
            for i in f.readlines():
                self.samples.append(i.replace('clean', 'final').strip().split(','))


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
#                  crop_type = 'random', crop_shape = None, resize_shape = None, resize_scale = None):
#         # super(DAVIS, self).__init__()
#         super().__init__()
#         self.color = color
#         self.crop_type = crop_type
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
#             crop_type = StaticRandomCrop(img1.shape[:2], self.crop_shape) if self.crop_type == 'random'\
#               else StaticCenterCrop(img1.shape[:2], self.crop_shape)
#             images = list(map(crop_type, images))

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
    """ 240fps high frame-rate dataset """
    def __init__(self, dataset_dir, train_or_val,
                 strides = 8, stretchable = False, origin_size = None,
                 crop_type = 'random', crop_shape = (384, 448),
                 resize_shape = None, resize_scale = None):
        super().__init__(dataset_dir, train_or_val, strides, stretchable,
                         origin_size, crop_type, crop_shape,
                         resize_shape, resize_scale)

    def has_no_txt(self):
        p = Path(self.dataset_dir)

        print('Collecting NeedforSpeed image paths, this operation can take a few minutes ...')
        collections_of_scenes = sorted(map(str, p.glob('**/240/*/*.jpg')))
        collections = [list(g) for k, g in groupby(collections_of_scenes, lambda x: x.split('/')[-2])]
        
        samples = [i for collection in collections for i in window(collection, self.strides)]
        self.split(samples)
        
def get_dataset(dataset_name):
    """
    Get specified dataset 
    Args: dataset_name str: target dataset name
    Returns: dataset tf.data.Dataset: target dataset class

    Available dataset pipelines
    - DAVIS
    - Sintel, SintelClean, SintelFinal
    - NeedforSpeed
    """
    datasets = {"DAVIS":DAVIS,
                "Sintel":Sintel,
                "SintelClean":SintelClean,
                "SintelFinal":SintelFinal,
                "NeedforSpeed":NeedforSpeed}
    return datasets[dataset_name]
    
