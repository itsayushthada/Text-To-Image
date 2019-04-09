import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F


class Generator(nn.Module):
	def __init__(self, batch_size, img_size, z_dim, text_embed_dim, text_reduced_dim):
		super(Generator, self).__init__()

		self.img_size = img_size
		self.z_dim = z_dim
		self.text_embed_dim = text_embed_dim

		self.concat = nn.Linear(z_dim + text_reduced_dim, 64 * 8 * 4 * 4).cuda()
		self.text_reduced_dim = nn.Linear(text_embed_dim, text_reduced_dim).cuda()
		
		# Defining the generator network architecture
		self.d_net = nn.Sequential(
			nn.ReLU(),
			nn.ConvTranspose2d(512, 256, 4, 2, 1),
			nn.BatchNorm2d(256),
			nn.ReLU(),
			nn.ConvTranspose2d(256, 128, 4, 2, 1),
			nn.BatchNorm2d(128),
			nn.ReLU(),
			nn.ConvTranspose2d(128, 64, 4, 2, 1),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.ConvTranspose2d(64, 3, 4, 2, 1),
			nn.Tanh()
		).cuda()

	def forward(self, text, z):
		""" Given a caption embedding and latent variable z(noise), generate an image

		Arguments
		---------
		text : torch.FloatTensor
			Output of the skipthought embedding model for the caption
			text.size() = (batch_size, text_embed_dim)

		z : torch.FloatTensor
			Latent variable or noise
			z.size() = (batch_size, z_dim)

		--------
		Returns
		--------
		output : An image of shape (64, 64, 3)

		"""
		reduced_text = self.text_reduced_dim(text.cuda())  # (batch_size, text_reduced_dim)
		concat = torch.cat((reduced_text, z.cuda()), 1)  # (batch_size, text_reduced_dim + z_dim)
		concat = self.concat(concat)  # (batch_size, 64*8*4*4)
		concat = concat.view(-1, 4, 4, 64 * 8)  # (batch_size, 4, 4, 64*8)
		
		concat = concat.permute(0, 3, 1, 2) # (batch_size, 512, 4, 4)
		d_net_out = self.d_net(concat)  # (batch_size, 3, 64, 64)
		d_net_out = d_net_out.permute(0, 2, 3, 1) #(batch_size, 64, 64, 3)
		
		output = d_net_out / 2. + 0.5   # (batch_size, 64, 64, 3)

		return output
