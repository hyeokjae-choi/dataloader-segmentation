import os.path as osp
import numpy as np
import cv2
import json
from abc import ABCMeta
from skimage.io import imread

from data.utils import augment_img_mask, crop
from data.data_config import DATASET_CONFIG

"""
- root_dir/
    - image/
    - mask/
        - mask1/
    - imageset/
        - single_Xclass/
            - train.txt
            - validataion.txt
            - test.txt
    - annotation
        - single_Xclass.json
"""


class DataManager(metaclass=ABCMeta):

    def __init__(self, ds_name, crop_method, crop_size, resize_size, pad_size, augment, div, **kwargs):
        if div not in ['train', 'validation', 'test']:
            raise ValueError("Invalid name of loss metric ({}).".format(div))

        self.crop_method = crop_method
        self.crop_size = crop_size
        self.augment = augment

        # Find information of the dataset.
        if ds_name in DATASET_CONFIG.SUALAB.keys():
            ds_info = DATASET_CONFIG.SUALAB[ds_name]
            self.ds_dir, anno_fname = ds_info['ds_dir'], ds_info['anno_fname']
            imgset_name = ds_info.get('imgset_name', None)
            imgset_number = ds_info.get('imgset_number', None)
            wrong_annotation_list = ds_info.get('wrong_annotation_list', [])
        else:
            raise ValueError('Unexpected name of the dataset ({})'.format(ds_name))

        # Remove wrong annotated data from data list.
        self.data_names = self.load_names(imgset_name, imgset_number, div)
        for wrong_annotation in wrong_annotation_list:
            if wrong_annotation in self.data_names:
                self.data_names.remove(wrong_annotation)

        # Load annotation data.
        with open(osp.join(self.ds_dir, 'annotation', anno_fname)) as json_data:
            self.cls = json.loads(json_data.read())

        # Read original image and mask data.
        self.imgs, self.masks = self.load_image_mask(imgset_name, resize_size, pad_size)
        if self.crop_method == 'sliding':
            self.imgs, self.masks = self.load_sliding_patch(stride=0.5)

        self.weighted_idx = self.weighted_indexing()
        self.num_patch_per_epoch = self.compute_num_patch_per_epoch()

    def __getitem__(self, index):
        name = self.weighted_idx[index]['name']
        patch_type = self.weighted_idx[index]['patch_type']
        return self.load_data(name, patch_type)

    def __len__(self):
        return len(self.weighted_idx)

    def size(self):
        return len(self.data_names)

    def load_names(self, imgset_name, imgset_number, div):
        if div == 'test':
            imgset_fpath = osp.join(self.ds_dir, 'imageset', imgset_name, '{}{}.txt'.format(div, imgset_number))
        else:
            imgset_fpath = osp.join(self.ds_dir, 'imageset', imgset_name, '{}{}_1.txt'.format(div, imgset_number))

        filename_list = []
        with open(imgset_fpath, 'r') as fid:
            filename_list += fid.read().split('\n')
        if filename_list[-1] == '':
            filename_list = filename_list[:-1]
        return filename_list

    def load_image_mask(self, imgset_name, resize_size, pad_size):
        imgs, masks = {}, {}
        for name in self.data_names:
            img = imread(osp.join(self.ds_dir, 'image', name))
            if len(img.shape) == 2:
                img = np.expand_dims(img, axis=2)
                img = np.concatenate([img, img, img], axis=-1)

            # 'OK' data has no mask image
            if self.cls['images'][name]['class'] == [0]:
                mask = np.zeros(img.shape[:2], dtype=np.uint8)
            else:
                mask = imread(osp.join(self.ds_dir, 'mask', imgset_name, name))

            ''' Resize data '''
            if resize_size != (None, None):
                img = cv2.resize(img, resize_size)
                mask = cv2.resize(mask, resize_size)

            ''' Padding data '''
            if pad_size != (None, None):
                pad_w = int(pad_size[0] / 2)
                pad_h = int(pad_size[1] / 2)
                img = np.pad(img, ((pad_w, pad_w), (pad_h, pad_h), (0, 0)), 'constant')
                if len(mask.shape) == 2:
                    mask = np.pad(mask, ((pad_w, pad_w), (pad_h, pad_h)), 'constant', constant_values=(255, 255))
                else:
                    mask = np.pad(mask, ((pad_w, pad_w), (pad_h, pad_h), (0, 0)), 'constant', constant_values=(255, 255))

            imgs[name] = np.asarray(img, dtype=np.float32)
            masks[name] = self.set_seg_label(mask)
        return imgs, masks

    def load_sliding_patch(self, stride):
        h, w = self.crop_size[0], self.crop_size[1]
        crop_imgs, crop_masks = {}, {}
        for name, img in self.imgs.items():
            img_size = img.shape
            for x in range(0, img_size[1] - w, int(w * stride)):
                for y in range(0, img_size[0] - h, int(h * stride)):
                    crop_img = self.imgs[name][y:y + h, x:x + w, :]
                    crop_mask = self.masks[name][y:y + h, x:x + w, :]
                    crop_imgs['{}_{:04d}_{:04d}.png'.format(osp.splitext(name)[0], x, y)] = crop_img
                    crop_masks['{}_{:04d}_{:04d}.png'.format(osp.splitext(name)[0], x, y)] = crop_mask
        return crop_imgs, crop_masks

    # One-hot-encoding
    def set_seg_label(self, mask):
        num_classes = len(self.cls['classes'])
        label = np.zeros((mask.shape[0], mask.shape[1], num_classes), dtype=np.float32)
        for cls in range(num_classes):
            idx = np.where(mask == cls)
            label[idx[0], idx[1], cls] = 1.
        return label

    # patch type = 0 : normal patch
    # patch type = 1 : defect patch
    def weighted_indexing(self):
        weighted_list = []
        if self.crop_method == 'sliding':
            list_defect, list_normal = [], []
            for name, mask in self.masks.items():
                if np.where(mask[:, :, 0] != 1)[0].shape[0] > 0:
                    list_defect.append(name)
                else:
                    list_normal.append(name)

            ratio = len(list_normal) // len(list_defect) if len(list_normal) // len(list_defect) != 0 else 1

            weighted_list += ratio * [{'name': n, 'patch_type': 1} for n in list_defect]
            weighted_list += [{'name': n, 'patch_type': 0} for n in list_normal]

        else:  # If crop_method is random or None
            list_defect = [n for n in self.data_names if self.cls['images'][n]['class'] != [0]]
            list_normal = [n for n in self.data_names if self.cls['images'][n]['class'] == [0]]

            ratio = (len(list_defect) + len(list_normal)) // len(list_defect)

            weighted_list += ratio * [{'name': n, 'patch_type': 1} for n in list_defect]
            weighted_list += [{'name': n, 'patch_type': 0} for n in list_defect]
            weighted_list += [{'name': n, 'patch_type': 0} for n in list_normal]

        return weighted_list

    def compute_num_patch_per_epoch(self):
        if self.crop_method == 'sliding':
            return len(self.weighted_idx)

        # Set the number of patches per 1 epoch that is the same as the 'sliding' crop method
        elif self.crop_method == 'random':
            _, crop_masks = self.load_sliding_patch(stride=0.5)

            weighted_list = []
            list_defect, list_normal = [], []
            for name, mask in crop_masks.items():
                if np.where(mask[:, :, 0] != 1)[0].shape[0] > 0:
                    list_defect.append(name)
                else:
                    list_normal.append(name)

            ratio = len(list_normal) // len(list_defect) if len(list_normal) // len(list_defect) != 0 else 1

            weighted_list += ratio * [{'name': n, 'patch_type': 1} for n in list_defect]
            weighted_list += [{'name': n, 'patch_type': 0} for n in list_normal]
            return len(weighted_list)

        else:  # If crop_method is None
            return len(self.weighted_idx)

    def load_data(self, name, patch_type):
        img = self.imgs[name]
        mask = self.masks[name]

        if self.crop_method == 'random':
            img, _, mask = crop(img, patch_type, mask, self.crop_size, 0.9)

        if self.augment is not None:
            img, mask = augment_img_mask(img, mask, augment=self.augment)

        # img = np.transpose(img, [2, 0, 1])
        # mask = np.transpose(mask, [2, 0, 1])
        return img, mask
