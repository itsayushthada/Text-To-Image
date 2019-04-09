import os
import torch
from pickle import load, dump
import skipthoughts
import numpy as np
from torch.autograd import Variable
from torch.utils.data import Dataset

class Text2ImageDataset(Dataset):
  
	def __init__(self, img_dir = "Images/images0.pickle", cap_dir = "Captions/captions_list.pickle"):
		self.images = self.load_pickle(img_dir)
		self.captions = self.load_pickle(cap_dir)
		self.model = skipthoughts.load_model()

	def load_pickle(self, pickle_path):
		obj = 0
		root_path = "/content/drive/My Drive/NLP/"
		with open(root_path + pickle_path, "rb") as inputfile:
			obj = load(inputfile)  
		return obj

	def encode_captions(self, captions_list):
		fun = lambda x: skipthoughts.encode(self.model, x) 
		encoded_list = list(map(fun, captions_list))
		return encoded_list

	def read_image(self, img_idx):
		return self.images[img_idx]

	def false_image(self, img_idx):
		idx = np.random.randint(0, self.__len__())
		if (idx != img_idx):
			return self.images[idx]
		return self.false_image(img_idx)

	def __len__(self):
		return self.images.shape[0]

	def __getitem__(self, idx):
		sample = {}
		sample["true_imgs"] = torch.FloatTensor(self.read_image(idx))
		sample["false_imgs"] = torch.FloatTensor(self.false_image(idx))
		sample["true_embds"] = torch.FloatTensor(self.encode_captions([self.captions[idx]])[0][np.random.randint(0,5)])
		return sample
