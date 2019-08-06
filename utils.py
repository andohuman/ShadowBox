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
	def __init__(self, root_dir='data', split='train', transforms=None):
		
		self.root_dir = root_dir
		self.img_files = glob(os.path.join(root_dir,split,'*.jpg'))
		self.all_combinations = list(itertools.combinations(self.img_files, 2))
		
		if transforms is None:
			self.transforms = transforms.Compose([
													transforms.RandomHorizontalFlip(),
													transforms.ToTensor()
				])

	def __len__(self):
		return len(self.all_combinations)

	def __getitem__(self, idx):

		img_file1, img_file2 = self.all_combinations[idx]

		img1, img2 = [cv2.imread(file,0) for file in [img_file1, img_file2]]
		img1_tensor, img2_tensor = [self.transforms(img) for img in [img1, img2]]

		if self.split == 'train':
			return [img1_tensor, img2_tensor], torch.tensor([1], dtype=torch.float) if img_file1.split('_')[0] == img_file2.split('_')[0] else torch.tensor([0], dtype=torch.float)
		elif self.split == 'valid':
			return [img1_tensor, img2_tensor], [img_file1, img_file2]



class _tqdm(tqdm):
	def format_num(self, n):
		f = '{:.5f}'.format(n)
		return f