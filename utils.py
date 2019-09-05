import cv2
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
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

class SiameseNet(nn.Module):
    def __init__(self, mode=None, weights_path=None, refs_dict=None, device=None):

        assert device is not None, "Specify a device to load the model into"
        assert mode in ['train', 'inference', 'validate'], "Unknown mode specified"

        super(SiameseNet, self).__init__()

        self.maxpool = nn.MaxPool2d(2)

        self.conv1 = nn.Conv2d(1, 64, 20)
        self.conv2 = nn.Conv2d(64, 128, 15)
        self.conv3 = nn.Conv2d(128, 128, 10)
        self.conv4 = nn.Conv2d(128, 256, 5)

        self.globalavgpool = nn.AdaptiveAvgPool2d((1,1))

        self.linear1 = nn.Linear(256, 128)
        self.linear2 = nn.Linear(128, 1)

        self.to(device) #push model to device

        if mode == 'inference':
            assert weights_path is not None or refs_dict is not None, "Please provide weights_path and reference dictionary"
            self.eval() #set to eval mode till ref features are computed
            self.load_state_dict(torch.load(weights_path, map_location=device))
            self.feature_list = [self.compute_feature(i) for i in refs_dict.values()] #compute ref features for inference
        
        elif mode == 'validate':
            assert weights_path is not None, "Please provide weights_path"
            self.load_state_dict(torch.load(weights_path, map_location=device))

        self.train() #set to train mode under any circumstance (for some reason, inference time in this mode is much faster)

    def compute_feature(self, data): #compute features of the input data

        x = self.conv1(data)
        x = F.relu(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.maxpool(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.maxpool(x)

        x = self.conv4(x)
        x = F.relu(x)
        x = self.globalavgpool(x)

        x = x.view(x.shape[0], -1)

        x = self.linear1(x)

        return x

    def classify(self, feature1, feature2): #check for similarity here

        x = torch.abs(feature2 - feature1)
        x = self.linear2(x)
        return torch.sigmoid(x)

    def forward(self, data1, data2=None):

        if data2 is not None: #train or valid mode

            features = [self.compute_feature(data) for data in [data1, data2]]
            prob = self.classify(features[0], features[1])

            return prob

        elif data2 is None: #inference mode

            feature = self.compute_feature(data1)

            return [self.classify(feature, i) for i in self.feature_list]

class _tqdm(tqdm):
    def format_num(self, n):
        f = '{:.5f}'.format(n)
        return f
