import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import random
import math

def mask_operate(data, mask, col_ind):
	result = []
	for i in range(mask.shape[1]):
		sta = col_ind[i][0]
		end = col_ind[i][1]
		#data[:, sta:end] = data[:, sta:end]*mask[:, i:i+1]
		result.append(data[:, sta:end]*mask[:, i:i+1])
	result = torch.cat(result, dim = 1)
	return result

def compute_kl(real, pred):
	return torch.sum((torch.log(pred + 1e-4) - torch.log(real + 1e-4)) * pred)

def KL_Loss(x_fake, x_real, col_type, col_dim):
	kl = 0.0
	sta = 0
	end = 0
	for i in range(len(col_type)):
		dim = col_dim[i]
		sta = end
		end = sta+dim
		fakex = x_fake[:,sta:end]
		realx = x_real[:,sta:end]
		if col_type[i] == "gmm":
			fake2 = fakex[:,1:]
			real2 = realx[:,1:]
			dist = torch.sum(fake2, dim=0)
			dist = dist / torch.sum(dist)
			real = torch.sum(real2, dim=0)
			real = real / torch.sum(real)
			kl += compute_kl(real, dist)
		else:
			dist = torch.sum(fakex, dim=0)
			dist = dist / torch.sum(dist)
			
			real = torch.sum(realx, dim=0)
			real = real / torch.sum(real)
			
			kl += compute_kl(real, dist)
	return kl



class Handler:
	def __init__(self, mask_gen=None, obs_gen=None, noise_gen=None, mask_dis=None, obs_dis=None, noise_dis=None, target_dis=None):
		self.mask_gen = mask_gen
		self.obs_gen = obs_gen
		self.noise_gen = noise_gen
		self.mask_dis = mask_dis
		self.obs_dis = obs_dis
		self.noise_dis = noise_dis
		self.target_dis = target_dis
		

	def mask_train(self, z_dim, epochs, steps_per_epoch, lr, dataloader, log_path=None, GPU=False):
		itertimes = steps_per_epoch/10
		torch.manual_seed(0)
		torch.cuda.manual_seed(0)
		torch.cuda.manual_seed_all(0)
		if GPU:
			self.mask_gen.cuda()
			self.mask_dis.cuda()

		G_optim = optim.SGD(self.mask_gen.parameters(), lr=lr, weight_decay=1e-5)
		D_optim = optim.SGD(self.mask_dis.parameters(), lr=lr, weight_decay=1e-5)
		
		for epoch in range(epochs):
			if log_path is not None:
				with open(log_path, "a+") as log:
					log.write("-----------Epoch {}-----------\n".format(epoch))
			print("-----------Epoch {}-----------".format(epoch))
			n_dis = 0
			for it in range(steps_per_epoch):
				''' train Discriminator '''
				x, c, m_real = dataloader.sample(mask=True)
				z = torch.randn(m_real.shape[0], z_dim)
				if GPU:
					z = z.cuda()
					m_real = m_real.cuda()

				m_fake = self.mask_gen(z)
				
				y_real = self.mask_dis(m_real)
				y_fake = self.mask_dis(m_fake)
				
				D_Loss = -(torch.mean(y_real) - torch.mean(y_fake))
				
				G_optim.zero_grad()
				D_optim.zero_grad()
				D_Loss.backward()
				D_optim.step()

				for p in self.mask_dis.parameters():
					p.data.clamp_(-0.1, 0.1)

				n_dis += 1

				if n_dis == 5:
					n_dis = 0
					''' train Generator '''
					z = torch.randn(m_real.shape[0], z_dim)
					if GPU:
						z = z.cuda()
					
					m_fake = self.mask_gen(z)
					y_fake = self.mask_dis(m_fake)
					
					G_Loss = -torch.mean(y_fake)

					G_optim.zero_grad()
					D_optim.zero_grad()
					G_Loss.backward()
					G_optim.step()

				if it>=5 and it%itertimes == 0:
					if log_path is not None:
						with open(log_path,"a+") as log:
							log.write("iterator {}, D_Loss:{}, G_Loss:{}\n".format(it, D_Loss.data, G_Loss.data))
					print("iterator {}, D_Loss:{}, G_Loss:{}\n".format(it, D_Loss.data, G_Loss.data))
		if GPU:
			self.mask_gen.cpu()
			self.mask_dis.cpu()	


	def obs_train(self, obs_gen, obs_dis, z_dim, epochs, steps_per_epoch, lr, dataloader, dataset, log_path=None, GPU=False):
		itertimes = steps_per_epoch/50
		torch.manual_seed(0)
		torch.cuda.manual_seed(0)
		torch.cuda.manual_seed_all(0)
		np.random.seed(0)
		if GPU:
			obs_gen.GPU = True
			obs_gen.cuda()
			obs_dis.cuda()

		D_optim = optim.SGD(obs_dis.parameters(), lr=lr, weight_decay=1e-5)
		G_optim = optim.SGD(obs_gen.parameters(), lr=lr, weight_decay=1e-5)
		all_labels = dataloader.label
		conditions = np.unique(all_labels.view(all_labels.dtype.descr * all_labels.shape[1]))
		for epoch in range(epochs):
			if log_path is not None:
				with open(log_path, "a+") as log:
					log.write("-----------Epoch {}-----------\n".format(epoch))
			print("-----------Epoch {}-----------".format(epoch))
			#n_dis = 0
			for it in range(steps_per_epoch):
				c = random.choice(conditions)
				c_ = c
				while c_ == c:
					c_ = random.choice(conditions) 

				''' train Discriminator '''
				x_real, c_real, m_real = dataloader.sample(label=list(c), mask=True)
				x_real_, c_real_, m_real_ = dataloader.sample(label=list(c_), mask=True)
				x_real = mask_operate(x_real, m_real, dataset.col_ind)
				x_real_ = mask_operate(x_real_, m_real_, dataset.col_ind)

				z = torch.randn(x_real.shape[0], z_dim)
				if GPU:
					z = z.cuda()
					x_real = x_real.cuda()
					x_real_ = x_real_.cuda()
					c_real = c_real.cuda()
					m_real = m_real.cuda()

				x_fake = obs_gen(z, c_real)
				x_fake = mask_operate(x_fake, m_real, dataset.col_ind)
				
				
				y_real = obs_dis(x_real, c_real)
				y_real_ = obs_dis(x_real_, c_real)
				y_fake = obs_dis(x_fake, c_real)
				
				fake_label = torch.zeros(y_fake.shape[0], 1)
				real_label = np.ones([y_real.shape[0], 1])
				real_label = real_label * 0.7 + np.random.uniform(0, 0.3, real_label.shape)
				real_label = torch.from_numpy(real_label).float()
				if GPU:
					fake_label = fake_label.cuda()
					real_label = real_label.cuda()

				D_Loss1 = F.binary_cross_entropy(y_real, real_label)
				D_Loss2 = F.binary_cross_entropy(y_fake, fake_label)
				D_Loss3 = F.binary_cross_entropy(y_real_, fake_label)
				D_Loss = D_Loss1 + D_Loss2 + D_Loss3
				
				G_optim.zero_grad()
				D_optim.zero_grad()
				D_Loss.backward()
				D_optim.step()

				#for p in obs_dis.parameters():
				#	p.data.clamp_(-0.1, 0.1)

				#n_dis += 1
				
				#if n_dis == 2:
					#n_dis = 0
				''' train Generator '''
				z = torch.randn(x_real.shape[0], z_dim)
				if GPU:
					z = z.cuda()
				
				x_fake = obs_gen(z, c_real)
				x_fake = mask_operate(x_fake, m_real, dataset.col_ind)
				y_fake = obs_dis(x_fake, c_real)
				
				real_label = torch.ones(y_fake.shape[0], 1)
				if GPU:
					real_label = real_label.cuda()
				G_Loss1 = F.binary_cross_entropy(y_fake, real_label)
				if KL:
					KL_loss = KL_Loss(x_fake, x_real, col_type, dataset.col_dim)
					G_Loss = G_Loss1 + KL_loss
				else:
					G_Loss = G_Loss1

				G_optim.zero_grad()
				D_optim.zero_grad()
				G_Loss.backward()
				G_optim.step()

				if it>=5 and it%itertimes == 0:
					if log_path is not None:
						with open(log_path,"a+") as log:
							log.write("iterator {}, D_Loss:{}, G_Loss:{}\n".format(it, D_Loss.data, G_Loss.data))
					print("iterator {}, D_Loss:{}, G_Loss:{}\n".format(it, D_Loss.data, G_Loss.data))

			#self.sample(z_dim, dataloader.label.shape[0], dataloader, maskloader, dataset, self.path+"sample_{}_{}".format())
		if GPU:
			obs_gen.GPU = False
			obs_gen.cpu()
			obs_dis.cpu()
		return obs_gen, obs_dis

	def noise_train():
		pass

	def joint_train(self, obs_gen, target_dis, zm_dim, zx_dim, epochs, steps_per_epoch, lr, sourceloader, targetloader, dataset, log_path=None, GPU=False):
		itertimes = steps_per_epoch/50
		torch.manual_seed(0)
		torch.cuda.manual_seed(0)
		torch.cuda.manual_seed_all(0)
		np.random.seed(0)
		if GPU:
			#self.mask_gen.cuda()
			obs_gen.cuda()
			target_dis.cuda()

		#Gm_optim = optim.SGD(self.mask_gen.parameters(), lr=lr, weight_decay=1e-5)
		Go_optim = optim.SGD(obs_gen.parameters(), lr=lr, weight_decay=1e-5)
		D_optim = optim.SGD(target_dis.parameters(), lr=lr, weight_decay=1e-5)
		
		for epoch in range(epochs):
			if log_path is not None:
				with open(log_path, "a+") as log:
					log.write("-----------Epoch {}-----------\n".format(epoch))
			print("-----------Epoch {}-----------".format(epoch))
			n_dis = 0
			for it in range(steps_per_epoch):
				''' train Discriminator '''
				obs_x, obs_c, obs_m = sourceloader.sample(mask=True)
				x_real, c_real, m_real = targetloader.sample(mask=True)
				x_real = mask_operate(x_real, m_real, dataset.col_ind)

				zm = torch.randn(x_real.shape[0], zm_dim)
				zx = torch.randn(x_real.shape[0], zx_dim)
				if GPU:
					zm = zm.cuda()
					zx = zx.cuda()
					obs_c = obs_c.cuda()
					x_real = x_real.cuda()
					m_real = m_real.cuda()

				#m_fake = self.mask_gen(zm)
				obs_fake = obs_gen(zx, obs_c)
				x_fake = mask_operate(obs_fake, m_real, dataset.col_ind)
				#m_fake = torch.round(m_fake)
				#x_fake = obs_fake
				#x_fake = mask_operate(obs_fake, m_fake, dataset.col_ind)
				
				y_real = target_dis(x_real)
				y_fake = target_dis(x_fake)
					
				D_Loss = -(torch.mean(y_real) - torch.mean(y_fake))
				
				#Gm_optim.zero_grad()
				Go_optim.zero_grad()
				D_optim.zero_grad()
				D_Loss.backward()
				D_optim.step()

				for p in target_dis.parameters():
					p.data.clamp_(-0.1, 0.1)

				n_dis += 1

				if n_dis == 5:
					n_dis = 0
					''' train Generator '''
					zm = torch.randn(x_real.shape[0], zm_dim)
					zx = torch.randn(x_real.shape[0], zx_dim)
					if GPU:
						zm = zm.cuda()
						zx = zx.cuda()
					
					#m_fake = self.mask_gen(zm)
					obs_fake = obs_gen(zx, obs_c)
					x_fake = mask_operate(obs_fake, m_real, dataset.col_ind)
					#m_fake = torch.round(m_fake)
					#x_fake = obs_fake
					#x_fake = mask_operate(obs_fake, m_fake, dataset.col_ind)

					y_fake = target_dis(x_fake)

					G_Loss = -torch.mean(y_fake)

					#Gm_optim.zero_grad()
					Go_optim.zero_grad()
					D_optim.zero_grad()
					G_Loss.backward()
					#Gm_optim.step()
					Go_optim.step()

				if it>=5 and it%itertimes == 0:
					if log_path is not None:
						with open(log_path,"a+") as log:
							log.write("iterator {}, D_Loss:{}, G_Loss:{}\n".format(it,D_Loss.data, G_Loss.data))
					print("iterator {}, D_Loss:{}, G_Loss:{}\n".format(it,D_Loss.data, G_Loss.data))

			#self.sample(zm_dim, zx_dim, sourceloader.label.shape[0], sourceloader, dataset, path+"sample_{}_{}".format())

		if GPU:
			#self.mask_gen.cpu()
			obs_gen.cpu()
			target_dis.cpu()
		return obs_gen, target_dis

	def save(self, mname, mpath):
		torch.save(self.__dict__[mname], mpath)

	def load(self, mname, mpath):
		self.__dict__[mname] = torch.load(mpath)

	def sample(self, obs_gen, zm_dim, zx_dim, rows, sourceloader, targetloader, dataset, path, GPU=True, repeat=1):
		self.obs_gen.eval()
		#self.mask_gen.eval()

		if GPU:
			self.obs_gen.cuda()
			#self.mask_gen.cuda()

		for time in range(repeat):
			sample_data = []
			while len(sample_data) < rows:
				x_source, c_source, m_source = sourceloader.sample(mask=True)
				x_target, c_target, m_target = targetloader.sample(mask=True)
				zx = torch.randn(x_source.shape[0], zx_dim)
				zm = torch.randn(x_source.shape[0], zm_dim)

				if GPU:
					c_source = c_source.cuda()
					zx = zx.cuda()
					zm = zm.cuda()

				obs_fake = self.obs_gen(zx, c_source)
				#m_fake = self.mask_gen(zm)
				#m_fake = torch.round(m_fake)
				#m_fake = m_fake.cpu()
				#m_fake = m_fake.detach().numpy()
				m_target = m_target.detach().numpy()

				samples = torch.cat((obs_fake, c_source), dim=1)
				samples = samples.reshape(samples.shape[0], -1)
				samples = samples[:,:dataset.dim]
				samples = samples.cpu()
				sample_table = dataset.reverse(samples.detach().numpy())
				#m_fake = np.concatenate([m_fake, np.ones([m_fake.shape[0], sample_table.shape[1]-m_fake.shape[1]])], axis=1)
				m_target = np.concatenate([m_target, np.ones([m_target.shape[0], sample_table.shape[1]-m_target.shape[1]])], axis=1)
				for i in range(sample_table.shape[0]):
					for j in range(sample_table.shape[1]):
						if m_target[i][j] == 0:
							sample_table[i][j] = 0
				df = pd.DataFrame(sample_table, columns=dataset.columns)
				if len(sample_data) == 0:
					sample_data = df
				else:
					sample_data = sample_data.append(df)
			sample_data = sample_data.iloc[:rows]
			sample_data.to_csv(path+"_"+str(time)+".csv", index = None)
		
		self.obs_gen.train()
		#self.mask_gen.train()
		
		
		
		
