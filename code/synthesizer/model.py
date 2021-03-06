import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
import pandas as pd
import os
from torch.autograd import Variable

def mask_operate(data, mask, col_ind):
	result = []
	for i in range(mask.shape[1]):
		sta = col_ind[i][0]
		end = col_ind[i][1]
		#data[:, sta:end] = data[:, sta:end]*mask[:, i:i+1]
		result.append(data[:, sta:end]*mask[:, i:i+1])
	result = torch.cat(result, dim = 1)
	return result

class MaskGenerator_MLP(nn.Module):
	def __init__(self, z_dim, data_dim, hidden_dims, mask_dim):
		super(MaskGenerator_MLP, self).__init__()
		dim = z_dim+data_dim
		seq = []
		for item in list(hidden_dims):
			fc = nn.Linear(dim, item)
			nn.init.xavier_normal_(fc.weight)
			seq += [
				fc,
				nn.BatchNorm1d(item),
				nn.Dropout(0.5),
				nn.LeakyReLU()
			]
			dim = item
		seq += [nn.Linear(dim, mask_dim)]
		nn.init.xavier_normal_(seq[-1].weight)
		self.net = nn.Sequential(*seq)

	def forward(self, z, x):
		z = torch.cat((z,x),dim=1)
		m = self.net(z)
		m = torch.sigmoid(m)
		return m


class ObservedGenerator_MLP(nn.Module):
	def __init__(self, z_dim, hidden_dims, x_dim, mask_dim, col_type, col_ind):
		super(ObservedGenerator_MLP, self).__init__()
		self.col_type = col_type
		self.col_ind = col_ind
		dim = z_dim + x_dim + mask_dim
		seq = []
		for item in list(hidden_dims):
			fc = nn.Linear(dim, item)
			nn.init.xavier_normal_(fc.weight)
			seq += [
				fc,
				nn.BatchNorm1d(item),
				nn.Dropout(0.5),
				nn.ReLU()
			]
			dim = item
		seq += [nn.Linear(dim, x_dim)]
		nn.init.xavier_normal_(seq[-1].weight)
		self.net = nn.Sequential(*seq)

	def forward(self, z, x, m):
		input = torch.cat((z, x, m), dim = 1)
		data = self.net(input)
		output = []
		for i in range(len(self.col_type)):
			sta = self.col_ind[i][0]
			end = self.col_ind[i][1]
			if self.col_type[i] == 'condition':
				continue
			if self.col_type[i] == 'normalize':
				temp = torch.tanh(data[:,sta:end])
			elif self.col_type[i] == 'one-hot':
				temp = torch.softmax(data[:,sta:end], dim=1)
			elif self.col_type[i] == 'gmm':
				temp1 = torch.tanh(data[:,sta:sta+1])
				temp2 = torch.softmax(data[:,sta+1:end], dim=1)
				temp = torch.cat((temp1,temp2),dim=1)
			elif self.col_type[i] == 'ordinal':
				temp = torch.sigmoid(data[:,sta:end])
			output.append(temp)
		output = torch.cat(output, dim = 1)
		return output

class ObservedGenerator_LSTM(nn.Module):
	def __init__(self, z_dim, feature_dim, lstm_dim, col_dim, col_type, col_ind, x_dim, mask_dim):
		super(ObservedGenerator_LSTM, self).__init__()
		self.x_dim = x_dim
		self.mask_dim = mask_dim
		self.l_dim = lstm_dim
		self.f_dim = feature_dim
		self.col_dim = col_dim
		self.col_ind = col_ind
		self.col_type = col_type
		self.GPU = False
		self.LSTM = nn.LSTMCell(z_dim+x_dim+mask_dim+feature_dim, lstm_dim) # input (fx, z, attention), output(hx, cx)
		self.FC = {}	 # FullyConnect layers for every columns 
		self.Feature = {}
		self.go = nn.Parameter(torch.randn(1, self.f_dim))
		for i in range(len(col_type)):
			if col_type[i] == "condition":
				continue
			if col_type[i] == "gmm":
				self.FC[i] = []
				fc1 = nn.Linear(feature_dim, 1)
				setattr(self, "gmfc%d0"%i, fc1)
				
				fc2 = nn.Linear(feature_dim, col_dim[i] - 1)
				setattr(self, "gmfc%d1"%i, fc2)
				
				fc3 = nn.Linear(col_dim[i] - 1, feature_dim)
				setattr(self, "gmfc%d2"%i, fc3)
				self.FC[i] = [fc1, fc2, fc3]
				
				fe1 = nn.Linear(lstm_dim, feature_dim)
				setattr(self, "gmfe%d0"%i, fe1)
				
				fe2 = nn.Linear(lstm_dim, feature_dim)
				setattr(self, "gmfe%d1"%i, fe2)
				self.Feature[i] = [fe1, fe2]
			else:
				fc1 = nn.Linear(feature_dim, col_dim[i])
				setattr(self, "fc%d0"%i, fc1)
				
				fc2 = nn.Linear(col_dim[i], feature_dim)
				setattr(self, "fc%d1"%i, fc2)
				self.FC[i] = [fc1, fc2]
				
				fe = nn.Linear(lstm_dim, feature_dim)
				setattr(self, "fe%d"%i, fe)
				self.Feature[i] = fe

	def forward(self, z, x, m):
		states = []
		outputs = []
		z = torch.cat((z, x, m),dim=1)
		hx = torch.randn(z.size(0), self.l_dim)
		cx = torch.randn(z.size(0), self.l_dim)
		fx = self.go.repeat([z.size(0), 1])
		if self.GPU:
			hx = hx.cuda()
			cx = cx.cuda()
			fx = fx.cuda()
		inputs = torch.cat((z, fx), dim = 1)
		for i in range(len(self.col_type)):
			if self.col_type[i] == "condition":
				continue
			if self.col_type[i] == "gmm":
				hx, cx = self.LSTM(inputs, (hx, cx))
				states.append(hx)
				fx = torch.tanh(self.Feature[i][0](hx))
				v = torch.tanh(self.FC[i][0](fx))
				outputs.append(v)
				inputs = torch.cat((z, fx), dim = 1)
				
				hx, cx = self.LSTM(inputs, (hx, cx))
				states.append(hx)
				fx = torch.tanh(self.Feature[i][1](hx))
				v = self.FC[i][1](fx)
				v = torch.softmax(v, dim=1)
				outputs.append(v)
				fx = torch.tanh(self.FC[i][2](v))
				inputs = torch.cat((z, fx), dim = 1)
			else:
				hx, cx = self.LSTM(inputs, (hx, cx))
				states.append(hx)
				fx = self.Feature[i](hx)
				v = self.FC[i][0](fx)
				if self.col_type[i] == "normalize":
					v = torch.tanh(v)
				elif self.col_type[i] == "one-hot":
					v = torch.softmax(v, dim = 1)
				elif self.col_type[i] == "ordinal":
					v = torch.sigmoid(v)
				outputs.append(v)
				fx = self.FC[i][1](v)
				inputs = torch.cat((z, fx), dim = 1)
		true_output = torch.cat(outputs, dim = 1)
		return true_output

class Discriminator(nn.Module):
	def __init__(self, x_dim, hidden_dims, c_dim=0, condition=False):
		super(Discriminator, self).__init__()
		self.condition = condition
		dim = x_dim+c_dim
		seq = []
		for item in list(hidden_dims):
			fc = nn.Linear(dim, item)
			nn.init.xavier_normal_(fc.weight)
			seq += [
				fc,
				nn.BatchNorm1d(item),
				nn.Dropout(0.5),
				nn.LeakyReLU()
			]
			dim = item
		seq += [nn.Linear(dim, 1)]
		nn.init.xavier_normal_(seq[-1].weight)
		self.net = nn.Sequential(*seq)

	def forward(self, x, c=None):
		if self.condition:
			x = torch.cat((x, c), dim=1)
		y = self.net(x)
		return y

	def init_weights(self):
		pass

class Noise_Discriminator(nn.Module):
	def __init__(self, x_dim, hidden_dims, mask_dim):
		super(Noise_Discriminator, self).__init__()
		dim = x_dim
		seq = []
		for item in list(hidden_dims):
			seq += [
				nn.Linear(dim, item),
				nn.BatchNorm1d(item),
				nn.Dropout(0.5),
				nn.LeakyReLU()
			]
			dim = item
		seq += [
			nn.Linear(dim, mask_dim),
			nn.Sigmoid()
		]
		self.net = nn.Sequential(*seq)

	def forward(self, x):
		y = self.net(x)
		return y
