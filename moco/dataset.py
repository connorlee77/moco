import torchvision.datasets as datasets
import glob
import os
import cv2
import albumentations as A 
from albumentations.pytorch import ToTensorV2


import numpy as np
import random
import cv2
import skimage
from skimage import exposure
from albumentations.core.transforms_interface import ImageOnlyTransform
from albumentations.augmentations.utils import (
    is_grayscale_image,
    is_rgb_image,
)


def loader(path):
    img = cv2.imread(path, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

class Grayscale(ImageOnlyTransform):

    def __init__(self, always_apply=False, p=0.5):
        super(Grayscale, self).__init__(always_apply, p)

    def apply(self, img, **params):
        img = skimage.color.rgb2gray(img)            
        return img


class SkimageCLAHE(ImageOnlyTransform):
    """Apply Contrast Limited Adaptive Histogram Equalization to the input image.

    Args:
        clip_limit (float or (float, float)): upper threshold value for contrast limiting.
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(self, clip_limit=0.025, random_clip=False, always_apply=False, p=0.5):
        super(SkimageCLAHE, self).__init__(always_apply, p)
        self.clip_limit = clip_limit
        self.random_clip = random_clip

    def apply(self, img, clip_limit=0.025, **params):
        if not is_rgb_image(img) and not is_grayscale_image(img):
            raise TypeError("CLAHE transformation expects 1-channel or 3-channel images.")

        img = exposure.equalize_adapthist(img, clip_limit=clip_limit)
        return img

    def get_params(self):
        clip_limit = self.clip_limit
        if self.random_clip:
            clip_limit = random.uniform(0.005, self.clip_limit)
        return {"clip_limit": clip_limit}

    def get_transform_init_args_names(self):
        return ("clip_limit")

class ThermalSSLDataset(datasets.ImageFolder):
    def __init__(self, root):
        super(ThermalSSLDataset, self).__init__(root, loader=loader)

        self.initial_crop = A.Compose([
            A.RandomCrop(256, 256),
        ])

        self.transform = A.Compose([
            A.Rotate(10, p=0.5),
            A.RandomResizedCrop(224, 224, scale=(0.75, 1.0)),
            A.HorizontalFlip(),
            A.OneOf([
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                SkimageCLAHE(random_clip=True),
            ], p=0.8),
            A.GaussianBlur(sigma_limit=[0.1, 2.0], p=0.5),
            Grayscale(p=1),
            A.Normalize(mean=0.5, std=0.25, max_pixel_value=1),
            ToTensorV2(),
        ])

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        
        h, w, c = sample.shape
        min_dim = min(h, w)
        if min_dim > 256:
            sample = A.SmallestMaxSize(random.randint(256, min_dim), p=0.5)(image=sample)['image']

        sample = np.float32(sample) / 255.0
        crop = self.initial_crop(image=sample)['image']
        sample1 = self.transform(image=crop)['image']
        sample2 = self.transform(image=crop)['image']
        
        return [sample1, sample2], target
