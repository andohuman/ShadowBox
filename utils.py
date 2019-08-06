import cv2
import numpy as numpy
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
from glob import glob
import os
import itertools
from tqdm import tqdm
from  torchvision import transforms

class Pairloader(Dataset):
	def __init__(self, root_dir='data', split='train', transform=None):
		
		self.root_dir = root_dir
		self.split = split
		self.img_files = glob(os.path.join(root_dir,split,'*.jpg'))
		self.all_combinations = list(itertools.combinations(self.img_files, 2))
		
		if transform is None:
			self.transform = transforms.Compose([
													transforms.ToTensor()
				])

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

class SiameseNet(nn.Module):
	def __init__(self):
		super(SiameseNet, self).__init__()

		self.pool = nn.MaxPool2d(2)
		self.conv1 = nn.Conv2d(1, 64, 10)
		self.conv2 = nn.Conv2d(64, 128, 7)
		self.conv3 = nn.Conv2d(128, 128, 4)
		self.conv4 = nn.Conv2d(128, 256, 4)
		self.linear1 = nn.Linear(239616, 1024)
		self.linear2 = nn.Linear(1024, 1)

	def forward(self, data):
		res = []

		for i in [0,1]:
			x = data[i]

			x = self.conv1(x)
			x = F.relu(x)
			x = self.pool(x)

			x = self.conv2(x)
			x = F.relu(x)
			x = self.pool(x)

			x = self.conv3(x)
			x = F.relu(x)
			x = self.pool(x)

			x = self.conv4(x)
			x = F.relu(x)
			x = self.pool(x)

			x = x.view(x.shape[0], -1)
			x = self.linear1(x)

			res.append(x)


		res = torch.abs(res[1] - res[0])
		res = self.linear2(res)
		res = torch.sigmoid(res)

		return res




