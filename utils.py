import cv2
import torch
from torch.utils.data import Dataset
import torch.nn as nn
from glob import glob
import os
import itertools
from tqdm import tqdm
from torchvision import transforms, models

class Pairloader(Dataset):
    def __init__(self, root_dir='data', split=None, transform=None):
        
        self.root_dir = root_dir
        self.split = split
        self.img_files = glob(os.path.join(root_dir,split,'*.jpg'))
        self.all_combinations = list(itertools.combinations(self.img_files, 2))
        
        if transform is None:
            self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.all_combinations)

    def __getitem__(self, idx):

        img_file1, img_file2 = self.all_combinations[idx]

        img1, img2 = [cv2.imread(file,0)[...,None] for file in [img_file1, img_file2]]
        img1_tensor, img2_tensor = [self.transform(img) for img in [img1, img2]]

        if self.split == 'train':
            return [img1_tensor, img2_tensor], torch.tensor([1], dtype=torch.float) if img_file1.split('_')[0] == img_file2.split('_')[0] else torch.tensor([0], dtype=torch.float)
        elif self.split == 'valid':
            return [img1_tensor, img2_tensor], [img_file1, img_file2]


class _tqdm(tqdm):
    def format_num(self, n):
        f = '{:.5f}'.format(n)
        return f

class Identity(nn.Module):
     def __init__(self):
         super(Identity, self).__init__()
     def forward(x):
        return x

class SiameseNet(nn.Module):
    def __init__(self):
         super(SiameseNet, self).__init__()
         squeezenet = models.squeezenet1_1(pretrained=True)
         squeezenet.features[0] = nn.Conv2d(1, 64, kernel_size=(3,3), stride=(2,2))
         squeezenet.fc = Identity()

         self.squeezenet = squeezenet
         self.linear = nn.Linear(1000, 1)
                                
    def forward(self, data):
        res = []

        for i in [0,1]:
            x = self.squeezenet(data[i])
            x = x.view(x.size(0), -1)
            res.append(x)

        res = torch.abs(res[1] - res[0])
        res = self.linear(res)
        res = torch.sigmoid(res)

        return res
