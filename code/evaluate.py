import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import sys
import argparse
import json
import multiprocessing
import os
import time
import logging
from pandas.api.types import is_object_dtype
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"

def feature_encoder(label_data):
	# transform categorical columns into numerical columns
	from sklearn.preprocessing import LabelEncoder
	from pandas.api.types import is_object_dtype
	label_con_data = label_data.copy()
	gens = {}
	for column in label_data.columns:
		if is_object_dtype(label_data[column]):
			gen_le = LabelEncoder()
			gen_labels = gen_le.fit_transform(list(label_data[column]))
			label_con_data.loc[:, column] = gen_labels  # to label from 0
			gens[column] = gen_le  # save the transformer to inverse
	# return a DataFrame
	return label_con_data, gens

class MLP(nn.Module):
	def __init__(self, shape, num_class):
		super(MLP, self).__init__()
		self.num_class = num_class
		output_shape = 1 if num_class == 2 else num_class
		self.net = nn.Sequential(
			nn.Linear(shape, 100),
			nn.BatchNorm1d(100),
			nn.ReLU(),

			nn.Linear(100, 50),
			nn.BatchNorm1d(50),
			nn.ReLU(),

			nn.Linear(50, output_shape)
		)

	def forward(self, x):
		y = self.net(x)
		if self.num_class == 2:
			y = torch.sigmoid(y)
		elif self.num_class > 2:
			y = torch.softmax(y, dim=1)
		return y

def train_simple(model, train_x, train_y, test_x, test_y, valid_x, valid_y, l2, epochs, lr, num_class):
	model.train()
	optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
	mlp_vals = []
	for epoch in range(epochs):
		if num_class == 2:
			y_ = model(train_x).squeeze()
			loss = F.binary_cross_entropy(y_, train_y)
		elif num_class > 2:
			y_ = model(train_x)
			loss = F.cross_entropy(y_, train_y.long())

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if epoch > 0 and epoch % 200 == 0:
			model.eval()
			print("iterator {}, Loss:{}".format(epoch, loss.data))
			logging.info("iterator {}, Loss:{}".format(epoch, loss.data))

			mlp_prob_valid_y = model(valid_x)
			if num_class == 2:
				mlp_pred_valid_y = (mlp_prob_valid_y > 0.5) + 0
			elif num_class > 2:
				mlp_pred_valid_y = torch.argmax(mlp_prob_valid_y, dim=1)
			mlp_pred_valid_y = mlp_pred_valid_y.cpu()
			mlp_val_valid = evaluation(valid_y.cpu(), mlp_pred_valid_y.cpu(), mlp_prob_valid_y.cpu(), num_class)

			mlp_prob_test_y = model(test_x)
			if num_class == 2:
				mlp_pred_test_y = (mlp_prob_test_y > 0.5) + 0
			elif num_class > 2:
				mlp_pred_test_y = torch.argmax(mlp_prob_test_y, dim=1)
			mlp_pred_test_y = mlp_pred_test_y.cpu()
			mlp_val_test = evaluation(test_y.cpu(), mlp_pred_test_y.cpu(), mlp_prob_test_y.cpu(), num_class)
			
			mlp_vals.append(["l2={},epoch={}".format(l2, epoch)]+mlp_val_valid+mlp_val_test)
			model.train()
	model.eval()
	return model, mlp_vals


def evaluation(y_true, y_pred, y_prob, num_class, threshold=None, parameters=None):
	from sklearn import metrics
	if num_class == 2:
		average = "binary"
	elif num_class > 2:
		average = "weighted"
	f1_score = metrics.f1_score(y_true, y_pred, average=average)  
	precision = metrics.precision_score(y_true, y_pred, average=average)
	recall = metrics.recall_score(y_true, y_pred, average=average)
	return [f1_score]


def train_classifiers_simple(train_data, test_data, label_column):
	torch.manual_seed(0)
	torch.cuda.manual_seed(0)
	torch.cuda.manual_seed_all(0)
	np.random.seed(0)
	#test_data = test_data.drop(test_data.columns[0:len(test_data.columns)-15], axis=1)
	test_data.columns = train_data.columns
	data = pd.concat([train_data, test_data], keys=['train', 'test'])
	num_class = len(data[label_column].unique())
	for i, col in enumerate(data.columns):
		if is_object_dtype(data[col]):
			data[col] = data[col].fillna(0.0)
		else:
			data[col] = data[col].fillna(data[col].mean())

	featured_con_data, gens = feature_encoder(data)

	label_column = label_column
	train_data = featured_con_data.loc['train']
	test_data = featured_con_data.loc['test']
	X_train = train_data.drop(axis=1, columns=[label_column])
	train_Y = train_data[label_column]
	X_test = test_data.drop(axis=1, columns=[label_column])
	test_Y = test_data[label_column]

	from sklearn import preprocessing
	length = int(len(X_train)/4)

	scaled_test_x = preprocessing.scale(X_test)
	scaled_train_x = preprocessing.scale(X_train.iloc[:length*3,:])
	scaled_valid_x = preprocessing.scale(X_train.iloc[length*3:,:])
	train_y = train_Y.iloc[:length*3]
	valid_y = train_Y.iloc[length*3:]

	print(scaled_valid_x.shape, scaled_train_x.shape, scaled_test_x.shape)

	scaled_valid_x = torch.from_numpy(scaled_valid_x).float().cuda()
	scaled_train_x = torch.from_numpy(scaled_train_x).float().cuda()
	train_y = torch.from_numpy(train_y.values).float().cuda()
	valid_y = torch.from_numpy(valid_y.values).float().cuda()
	scaled_test_x = torch.from_numpy(scaled_test_x).float().cuda()
	test_y = torch.from_numpy(test_Y.values).float().cuda()

	mlp_vals = []
	for l2 in [0,0.001,0.01,0.1]:	
		model = MLP(scaled_train_x.shape[1], num_class)
		model.cuda()
		model, mlp_val = train_simple(model, scaled_train_x, train_y, scaled_test_x, test_y, scaled_valid_x, valid_y, l2, 2000, 0.01, num_class)
		if len(mlp_vals) == 0:
			mlp_vals = mlp_val
		else:
			mlp_vals = mlp_vals + mlp_val
	return mlp_vals

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--train', type=str, default="tranlate.csv")
	parser.add_argument('--test', type=str, default="test.csv")
	parser.add_argument('--label_col', type=str, default="label")
	parser.add_argument('--output', type=str, default="result")
	parser.add_argument('--device', type=int, default=0)
	args = parser.parse_args()
	gpu = int(args.device)
	train = args.train
	test = args.test
	label_col = args.label_col
	output = args.output

	torch.manual_seed(0)
	torch.cuda.manual_seed(0)
	torch.cuda.manual_seed_all(0)
	np.random.seed(0)

	if gpu >= 0:
		GPU = True
		os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
	else:
		GPU = False

	test_data = pd.read_csv(test)
	train_data = pd.read_csv(train)

	logging.basicConfig(filename=output+'_log.log', level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT)
	mlp_vals = train_classifiers_simple(train_data, test_data, label_col)

	df = pd.DataFrame(mlp_vals, columns=["l2", "f1_valid", "f1_test"])
	df.to_csv(output+".csv", index=None)