### split file from FEAT

import os
from PIL import Image
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from .datasets import register
from PIL import ImageEnhance

tr
nsformtypedict=dict(Brightness=ImageEnhance.Brightness, Contrast=ImageEnhance.Contrast, Sharpness=ImageEnhance.Sharpness, Color=ImageEnhance.Color)


class ImageJitter(object):
    def __init__(self, transformdict):
        self.transforms = [(transformtypedict[k], transformdict[k]) for k in transformdict]

    def __call__(self, img):
        out = img
        randtensor = torch.rand(len(self.transforms))

        for i, (transformer, alpha) in enumerate(self.transforms):
            r = alpha*(randtensor[i]*2.0 -1.0) + 1
            out = transformer(out).enhance(r).convert('RGB')

        return out


@register('StanfordCars')
class Cars(Dataset):

    def __init__(self, root_path, split='train', **kwargs):

        split_tag = split
        txt_path = os.path.join(root_path, 'split', split_tag + '.csv')
        img_path = os.path.join(root_path, 'images')

        data, label = self.parse_csv(img_path, txt_path)

        image_size = 84
        data = [Image.open(x).convert('RGB').resize((92, 92)) for x in data] # image_size

        min_label = min(label)
        label = [x - min_label for x in label]

        self.data = data
        self.label = label
        self.n_classes = max(self.label) + 1

        norm_params = {'mean': [0.485, 0.456, 0.406],
                       'std': [0.229, 0.224, 0.225]}
        jitter_param = dict(Brightness=0.4, Contrast=0.4, Color=0.4)

        normalize = transforms.Normalize(**norm_params)
        self.default_transform = transforms.Compose([
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ])
        augment = kwargs.get('augment')
        if augment == 'resize':
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        elif augment == 'color':
            self.transform = transforms.Compose([
                transforms.Resize(image_size, Image.BILINEAR),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                normalize,
            ])
        elif augment == 'color3':
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                ImageJitter(jitter_param),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        elif augment == 'crop':
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.RandomCrop(image_size, padding=8),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        elif augment is None:
            self.transform = self.default_transform

        def convert_raw(x):
            mean = torch.tensor(norm_params['mean']).view(3, 1, 1).type_as(x)
            std = torch.tensor(norm_params['std']).view(3, 1, 1).type_as(x)
            return x * std + mean

        self.convert_raw = convert_raw

    def parse_csv(self, img_path, txt_path):
        data = []
        label = []
        lb = -1
        self.wnids = []
        lines = [x.strip() for x in open(txt_path, 'r').readlines()][1:]

        for l in lines:
            context = l.split(',')
            name = context[0]
            wnid = context[1]
            path = os.path.join(img_path, name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1

            data.append(path)
            label.append(lb)

        return data, label


    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.transform(self.data[i]), self.label[i]