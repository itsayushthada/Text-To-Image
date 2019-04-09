import os
import time
import datetime
import logging
import itertools

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision.utils import save_image

import numpy as np
from pickle import dump
from discriminator import Discriminator
from generator import Generator


class GAN_CLS(object):
	def __init__(self, args, data_loader, SUPERVISED=True):
		"""
		args : Arguments
		data_loader = An instance of class DataLoader for loading our dataset in batches
		"""

		self.data_loader = data_loader
		self.num_epochs = args.num_epochs
		self.batch_size = args.batch_size

		self.log_step = args.log_step
		self.sample_step = args.sample_step

		self.log_dir = args.log_dir
		self.checkpoint_dir = args.checkpoint_dir
		self.sample_dir = args.sample_dir
		self.final_model = args.final_model
		self.model_save_step = args.model_save_step

		#self.dataset = args.dataset
		#self.model_name = args.model_name

		self.img_size = args.img_size
		self.z_dim = args.z_dim
		self.text_embed_dim = args.text_embed_dim
		self.text_reduced_dim = args.text_reduced_dim
		self.learning_rate = args.learning_rate
		self.beta1 = args.beta1
		self.beta2 = args.beta2
		self.l1_coeff = args.l1_coeff
		self.resume_epoch = args.resume_epoch
		self.resume_idx = args.resume_idx
		self.SUPERVISED = SUPERVISED

		# Logger setting
		log_name = datetime.datetime.now().strftime('%Y-%m-%d')+'.log'
		self.logger = logging.getLogger('__name__')
		self.logger.setLevel(logging.INFO)
		self.formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
		self.file_handler = logging.FileHandler(os.path.join(self.log_dir, log_name))
		self.file_handler.setFormatter(self.formatter)
		self.logger.addHandler(self.file_handler)

		self.build_model()

	def smooth_label(self, tensor, offset):
		return tensor + offset
		
	def dump_imgs(images_Array, name):
		with open('{}.pickle'.format(name), 'wb') as file:
			dump(images_Array, file)
		
	def build_model(self):
		""" A function of defining following instances :

		-----  Generator
		-----  Discriminator
		-----  Optimizer for Generator
		-----  Optimizer for Discriminator
		-----  Defining Loss functions

		"""

		# ---------------------------------------------------------------------#
		#						1. Network Initialization					   #
		# ---------------------------------------------------------------------#
		self.gen = Generator(batch_size=self.batch_size,
							 img_size=self.img_size,
							 z_dim=self.z_dim,
							 text_embed_dim=self.text_embed_dim,
							 text_reduced_dim=self.text_reduced_dim)

		self.disc = Discriminator(batch_size=self.batch_size,
								  img_size=self.img_size,
								  text_embed_dim=self.text_embed_dim,
								  text_reduced_dim=self.text_reduced_dim)

		self.gen_optim = optim.Adam(self.gen.parameters(),
									lr=self.learning_rate,
									betas=(self.beta1, self.beta2))

		self.disc_optim = optim.Adam(self.disc.parameters(),
									 lr=self.learning_rate,
									 betas=(self.beta1, self.beta2))

		self.cls_gan_optim = optim.Adam(itertools.chain(self.gen.parameters(),
														self.disc.parameters()),
										lr=self.learning_rate,
										betas=(self.beta1, self.beta2))

		print ('-------------  Generator Model Info  ---------------')
		self.print_network(self.gen, 'G')
		print ('------------------------------------------------')

		print ('-------------  Discriminator Model Info  ---------------')
		self.print_network(self.disc, 'D')
		print ('------------------------------------------------')

		self.criterion = nn.BCELoss().cuda()
		# self.CE_loss = nn.CrossEntropyLoss().cuda()
		# self.MSE_loss = nn.MSELoss().cuda()
		self.gen.train()
		self.disc.train()

	def print_network(self, model, name):
		""" A function for printing total number of model parameters """
		num_params = 0
		for p in model.parameters():
			num_params += p.numel()

		print(model)
		print(name)
		print("Total number of parameters: {}".format(num_params))

	def load_checkpoints(self, resume_epoch, idx):
		"""Restore the trained generator and discriminator."""
		print('Loading the trained models from epoch {} and iteration {}...'.format(resume_epoch, idx))
		G_path = os.path.join(self.checkpoint_dir, '{}-{}-G.ckpt'.format(resume_epoch, idx))
		D_path = os.path.join(self.checkpoint_dir, '{}-{}-D.ckpt'.format(resume_epoch, idx))
		self.gen.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
		self.disc.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))

	def train_model(self):

		data_loader = self.data_loader

		start_epoch = 0
		if self.resume_epoch >= 0:
			start_epoch = self.resume_epoch
			self.load_checkpoints(self.resume_epoch, self.resume_idx)

		print ('---------------  Model Training Started  ---------------')
		start_time = time.time()

		for epoch in range(start_epoch, self.num_epochs):
			print("Epoch: {}".format(epoch+1))
			for idx, batch in enumerate(data_loader):
				print("Index: {}".format(idx+1), end = "\t")
				true_imgs = batch['true_imgs']
				true_embed = batch['true_embds']
				false_imgs = batch['false_imgs']

				real_labels = torch.ones(true_imgs.size(0))
				fake_labels = torch.zeros(true_imgs.size(0))

				smooth_real_labels = torch.FloatTensor(self.smooth_label(real_labels.numpy(), -0.1))

				true_imgs = Variable(true_imgs.float()).cuda()
				true_embed = Variable(true_embed.float()).cuda()
				false_imgs = Variable(false_imgs.float()).cuda()

				real_labels = Variable(real_labels).cuda()
				smooth_real_labels = Variable(smooth_real_labels).cuda()
				fake_labels = Variable(fake_labels).cuda()

				# ---------------------------------------------------------------#
				# 					  2. Training the generator                  #
				# ---------------------------------------------------------------#
				self.gen.zero_grad()
				z = Variable(torch.randn(true_imgs.size(0), self.z_dim)).cuda()
				fake_imgs = self.gen.forward(true_embed, z)
				fake_out, fake_logit = self.disc.forward(fake_imgs, true_embed)
				fake_out = Variable(fake_out.data, requires_grad=True).cuda()
				
				true_out, true_logit = self.disc.forward(true_imgs, true_embed)
				true_out = Variable(true_out.data, requires_grad=True).cuda()
				
				g_sf = self.criterion(fake_out, real_labels)
				#g_img = self.l1_coeff * nn.L1Loss()(fake_imgs, true_imgs)
				gen_loss = g_sf

				gen_loss.backward()
				self.gen_optim.step()

				# ---------------------------------------------------------------#
				# 					3. Training the discriminator				 #
				# ---------------------------------------------------------------#
				self.disc.zero_grad()
				false_out, false_logit = self.disc.forward(false_imgs, true_embed)
				false_out = Variable(false_out.data, requires_grad=True)
				
				sr = self.criterion(true_out, smooth_real_labels)
				sw = self.criterion(true_out, fake_labels)
				sf = self.criterion(false_out, smooth_real_labels)
				
				disc_loss =  torch.log(sr) + (torch.log(1-sw) + torch.log(1-sf ))/2 

				disc_loss.backward()
				self.disc_optim.step()

				self.cls_gan_optim.step()

				# Logging
				loss = {}
				loss['G_loss'] = gen_loss.item()
				loss['D_loss'] = disc_loss.item()

				# ---------------------------------------------------------------#
				# 					4. Logging INFO into log_dir				 #
				# ---------------------------------------------------------------#
				log = ""
				if (idx + 1) % self.log_step == 0:
					end_time = time.time() - start_time
					end_time = datetime.timedelta(seconds=end_time)
					log = "Elapsed [{}], Epoch [{}/{}], Idx [{}]".format(end_time, epoch + 1, self.num_epochs, idx)

				for net, loss_value in loss.items():
					log += "{}: {:.4f}".format(net, loss_value)
					self.logger.info(log)
					print (log)
				
				"""
				# ---------------------------------------------------------------#
				# 					5. Saving generated images					 #
				# ---------------------------------------------------------------#
				if (idx + 1) % self.sample_step == 0:
					concat_imgs = torch.cat((true_imgs, fake_imgs), 0)  # ??????????
					concat_imgs = (concat_imgs + 1) / 2
					# out.clamp_(0, 1)
					 
					save_path = os.path.join(self.sample_dir, '{}-{}-images.jpg'.format(epoch, idx + 1))
					# concat_imgs.cpu().detach().numpy()
					self.dump_imgs(concat_imgs.cpu().numpy(), save_path)
					
					#save_image(concat_imgs.data.cpu(), self.sample_dir, nrow=1, padding=0)
					print ('Saved real and fake images into {}...'.format(self.sample_dir))
				"""
				
				# ---------------------------------------------------------------#
				# 				6. Saving the checkpoints & final model			 #
				# ---------------------------------------------------------------#
				if (idx + 1) % self.model_save_step == 0:
					G_path = os.path.join(self.checkpoint_dir, '{}-{}-G.ckpt'.format(epoch, idx + 1))
					D_path = os.path.join(self.checkpoint_dir, '{}-{}-D.ckpt'.format(epoch, idx + 1))
					torch.save(self.gen.state_dict(), G_path)
					torch.save(self.disc.state_dict(), D_path)
					print('Saved model checkpoints into {}...\n'.format(self.checkpoint_dir))

		print ('---------------  Model Training Completed  ---------------')
		# Saving final model into final_model directory
		G_path = os.path.join(self.final_model, '{}-G.pth'.format('final'))
		D_path = os.path.join(self.final_model, '{}-D.pth'.format('final'))
		torch.save(self.gen.state_dict(), G_path)
		torch.save(self.disc.state_dict(), D_path)
		print('Saved final model into {}...'.format(self.final_model))
