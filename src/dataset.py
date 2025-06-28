import os
import glob
import scipy # Keep scipy for other potential uses, but not for imresize directly
import torch
import random
import numpy as np
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
from PIL import Image
from cv2 import imread # Changed from scipy.misc.imread
import cv2 # Import cv2 for imread and resize

from skimage.feature import canny
from skimage.color import rgb2gray, gray2rgb
from .utils import create_mask


class Dataset(torch.utils.data.Dataset):
    def __init__(self, config, flist, edge_flist, mask_flist, augment=True, training=True):
        super(Dataset, self).__init__()
        self.augment = augment
        self.training = training
        self.data = self.load_flist(flist)
        self.edge_data = self.load_flist(edge_flist)
        self.mask_data = self.load_flist(mask_flist)

        self.input_size = config.INPUT_SIZE
        self.sigma = config.SIGMA
        self.edge = config.EDGE
        self.mask = config.MASK
        self.nms = config.NMS

        # in test mode, there's a one-to-one relationship between mask and image
        # masks are loaded non random
        if config.MODE == 2:
            self.mask = 6

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        try:
            item = self.load_item(index)
        except Exception as e: # Catch a broader exception to log the error
            print(f'loading error for {self.data[index]}: {e}')
            item = self.load_item(0) # Fallback to item 0 if an error occurs

        return item

    def load_name(self, index):
        name = self.data[index]
        return os.path.basename(name)

    def load_item(self, index):

        size = self.input_size

        # load image
        img = imread(self.data[index]) # cv2.imread loads BGR by default

        # Convert BGR to RGB if needed by your model
        # Most PyTorch models expect RGB.
        if img is not None and len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


        # gray to rgb
        if len(img.shape) < 3:
            img = gray2rgb(img)

        # resize/crop if needed
        if size != 0:
            img = self.resize(img, size, size)

        # create grayscale image
        img_gray = rgb2gray(img)

        # load mask
        mask = self.load_mask(img, index)

        # load edge
        edge = self.load_edge(img_gray, index, mask)

        # augment data
        if self.augment and np.random.binomial(1, 0.5) > 0:
            img = img[:, ::-1, ...]
            img_gray = img_gray[:, ::-1, ...]
            edge = edge[:, ::-1, ...]
            mask = mask[:, ::-1, ...]

        return self.to_tensor(img), self.to_tensor(img_gray), self.to_tensor(edge), self.to_tensor(mask)

    def load_edge(self, img, index, mask):
        sigma = self.sigma

        # in test mode images are masked (with masked regions),
        # using 'mask' parameter prevents canny to detect edges for the masked regions
        # Ensure mask is boolean for canny if it's not None
        # Convert mask from 0-255 to boolean 0/1, then invert for canny's mask parameter
        mask_for_canny = None
        if not self.training and mask is not None:
             mask_for_canny = (mask == 0).astype(np.bool_) # Canny expects True for valid regions, False for masked.
                                                         # Assuming mask has 255 for missing regions, 0 for available.

        # canny
        if self.edge == 1:
            # no edge
            if sigma == -1:
                return np.zeros(img.shape).astype(np.float)

            # random sigma
            if sigma == 0:
                sigma = random.randint(1, 4)

            return canny(img, sigma=sigma, mask=mask_for_canny).astype(float)


        # external
        else:
            imgh, imgw = img.shape[0:2]
            edge = imread(self.edge_data[index]) # Uses cv2.imread

            # Handle grayscale image loading with cv2.imread
            if len(edge.shape) < 3:
                edge = gray2rgb(edge) # Convert to 3 channels if it's grayscale

            edge = self.resize(edge, imgh, imgw)

            # non-max suppression
            if self.nms == 1:
                # Need to ensure types match for multiplication.
                # Canny output is bool or float depending on version/arguments
                # Convert canny output to float if it's boolean
                canny_edges = canny(img, sigma=sigma, mask=mask_for_canny).astype(np.float)
                return edge * canny_edges # This expects edge to be float as well

            return edge

    def load_mask(self, img, index):
        imgh, imgw = img.shape[0:2]
        mask_type = self.mask

        # external + random block
        if mask_type == 4:
            mask_type = 1 if np.random.binomial(1, 0.5) == 1 else 3

        # external + random block + half
        elif mask_type == 5:
            mask_type = np.random.randint(1, 4)

        # random block
        if mask_type == 1:
            return create_mask(imgw, imgh, imgw // 2, imgh // 2)

        # half
        if mask_type == 2:
            # randomly choose right or left
            return create_mask(imgw, imgh, imgw // 2, imgh, 0 if random.random() < 0.5 else imgw // 2, 0)

        # external
        if mask_type == 3:
            mask_index = random.randint(0, len(self.mask_data) - 1)
            mask = imread(self.mask_data[mask_index]) # Uses cv2.imread
            # Handle grayscale image loading with cv2.imread
            if len(mask.shape) < 3:
                mask = gray2rgb(mask)
            mask = self.resize(mask, imgh, imgw)
            mask = (mask > 0).astype(np.uint8) * 255      # threshold due to interpolation
            return mask

        # test mode: load mask non random
        if mask_type == 6:
            mask = imread(self.mask_data[index]) # Uses cv2.imread
            # Handle grayscale image loading with cv2.imread
            if len(mask.shape) < 3:
                mask = gray2rgb(mask)
            mask = self.resize(mask, imgh, imgw, centerCrop=False)
            mask = rgb2gray(mask)
            mask = (mask > 0).astype(np.uint8) * 255
            return mask

    def to_tensor(self, img):
        # Ensure img is a numpy array (cv2.imread returns numpy arrays)
        # PIL.Image.fromarray expects a numpy array.
        # Ensure the image is in the correct type (e.g., uint8) for PIL
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        img = Image.fromarray(img)
        img_t = F.to_tensor(img).float()
        return img_t

    def resize(self, img, height, width, centerCrop=True):
        imgh, imgw = img.shape[0:2]

        if centerCrop and imgh != imgw:
            # center crop
            side = np.minimum(imgh, imgw)
            j = (imgh - side) // 2
            i = (imgw - side) // 2
            img = img[j:j + side, i:i + side, ...]

        # Use OpenCV's resize function instead of scipy.misc.imresize
        # INTER_AREA is good for shrinking, INTER_LINEAR for zooming.
        # Ensure img is numpy array for cv2.resize
        if not isinstance(img, np.ndarray):
            img = np.array(img)

        # If the image is grayscale but has 3 dimensions (e.g., (H, W, 1)), reshape to (H, W) for cv2.resize
        if len(img.shape) == 3 and img.shape[2] == 1:
            img = img.squeeze(axis=2)

        # For color images, cv2.resize works directly.
        # For grayscale images, if they were loaded as 3-channel, they're fine.
        # If they were naturally 1-channel, ensure it stays 1-channel after resize if desired,
        # or convert to 3-channel if the rest of the pipeline expects it.
        # Assuming the pipeline expects 3-channel after this point for consistency,
        # we'll ensure it's handled.
        if len(img.shape) == 2: # If it's a 2D grayscale image
            resized_img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
            # If the rest of the pipeline expects 3 channels, convert back
            resized_img = gray2rgb(resized_img)
        else: # For 3D images (RGB/BGR)
            resized_img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

        return resized_img

    def load_flist(self, flist):
        if isinstance(flist, list):
            return flist

        # flist: image file path, image directory path, text file flist path
        if isinstance(flist, str):
            if os.path.isdir(flist):
                flist = list(glob.glob(os.path.join(flist, '*.jpg'))) + \
                        list(glob.glob(os.path.join(flist, '*.png')))
                flist.sort()
                return flist

            if os.path.isfile(flist):
                try:
                    # Added encoding for better compatibility, and changed np.str to str
                    return np.genfromtxt(flist, dtype=str, encoding='utf-8')
                except:
                    return [flist]

        return []

    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True
            )

            for item in sample_loader:
                yield item