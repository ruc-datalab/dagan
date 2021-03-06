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


def consistency_loss(y_adapt, y_train, y):
	loss1 = F.binary_cross_entropy(y_train, y)
	loss2 = F.binary_cross_entropy(y_adapt, y) 
	loss3 = F.mse_loss(y_adapt, y_train) # 帮我confirm 下这里, OK
	return loss1, loss2, loss3


def no_consistency_loss(y_adapt, y_train, y):
	loss1 = F.binary_cross_entropy(y_train, y)
	loss2 = F.binary_cross_entropy(y_adapt, y)
	return loss1, loss2


class MLP(nn.Module):
	def __init__(self, shape, multiclass=False):
		super(MLP, self).__init__()
		self.net = nn.Sequential(
			nn.Linear(shape, 100),
			nn.BatchNorm1d(100),
			nn.ReLU(),

			nn.Linear(100, 50),
			nn.BatchNorm1d(50),
			nn.ReLU(),

			nn.Linear(50, 1),
			nn.Sigmoid()
		)

	def forward(self, x):
		return self.net(x)

def train_simple(model, train_x, train_y, test_xs, test_ys, valid_x, valid_y, l2, epochs, lr):
	model.train()
	optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
	mlp_vals = []
	mlp_loss = []
	for epoch in range(epochs):
		y_ = model(train_x)
		loss = F.binary_cross_entropy(y_, train_y)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		if epoch % 200 == 0:
			model.eval()
			print("iterator {}, Loss:{}".format(epoch, loss.data))
			logging.info("iterator {}, Loss:{}".format(epoch, loss.data))

			tmp_val = []
			tmp_loss = []
			mlp_prob_valid_y = model(valid_x)
			valid_loss = F.binary_cross_entropy(mlp_prob_valid_y, valid_y)
			mlp_pred_valid_y = (mlp_prob_valid_y > 0.5) + 0
			mlp_pred_valid_y = mlp_pred_valid_y.cpu()
			mlp_val_valid = evaluation(valid_y.cpu(), mlp_pred_valid_y.cpu(), mlp_prob_valid_y.cpu())
			tmp_val = tmp_val + mlp_val_valid
			tmp_loss = tmp_loss + [float(valid_loss.data)]

			for i in range(len(test_xs)):
				test_x = test_xs[i]
				test_y = test_ys[i]
				mlp_prob_test_y = model(test_x)
				test_loss = F.binary_cross_entropy(mlp_prob_test_y, test_y)
				mlp_pred_test_y = (mlp_prob_test_y > 0.5) + 0
				mlp_pred_test_y = mlp_pred_test_y.cpu()
				mlp_val_test = evaluation(test_y.cpu(), mlp_pred_test_y.cpu(), mlp_prob_test_y.cpu())  #mlp_thres, mlp_para)
				tmp_val = tmp_val + mlp_val_test
				tmp_loss = tmp_loss + [float(test_loss.data)]
			mlp_vals.append(["l2={},epoch={}".format(l2, epoch)]+tmp_val)
			mlp_loss.append(["l2={},epoch={}".format(l2, epoch)]+tmp_loss)
			model.train()
	model.eval()
	return model, mlp_vals, mlp_loss

def train_consistency(model, train_xs, train_ys, test_xs, test_ys, valid_x, valid_y, l2, epochs, lr):
	model.train()
	optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
	mlp_vals = []
	mlp_loss = []

	for epoch in range(epochs):
		loss_gdro = -999999999.0
		y_mean = 0.0
		for i in range(len(train_xs)):
			train_x = train_xs[i]
			train_y = train_ys[i]
			y_ = model(train_x)
			y_mean += y_
			loss = F.binary_cross_entropy(y_, train_y)
			if loss > loss_gdro:
				loss_gdro = loss
		y_mean = y_mean/len(train_xs)
		y_mean = y_mean.detach()

		loss_consistency = 0.0
		for i in range(len(train_xs)):
			train_x = train_xs[i]
			y_ = model(train_x)
			loss_consistency = loss_consistency + F.binary_cross_entropy(y_, y_mean)
		loss_consistency = loss_consistency/len(train_xs)

		loss = loss_gdro #+ 0.1*loss_consistency
		
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if epoch % 200 == 0:
			model.eval()
			print("iterator {}, Loss:{}".format(epoch, loss.data))
			logging.info("iterator {}, Loss:{}".format(epoch, loss.data))

			tmp_val = []
			tmp_loss = []
			mlp_prob_valid_y = model(valid_x)
			valid_loss = F.binary_cross_entropy(mlp_prob_valid_y, valid_y)
			mlp_pred_valid_y = (mlp_prob_valid_y > 0.5) + 0
			mlp_pred_valid_y = mlp_pred_valid_y.cpu()
			mlp_val_valid = evaluation(valid_y.cpu(), mlp_pred_valid_y.cpu(), mlp_prob_valid_y.cpu())
			tmp_val = tmp_val + mlp_val_valid
			tmp_loss = tmp_loss + [float(valid_loss.data)]

			for i in range(len(test_xs)):
				test_x = test_xs[i]
				test_y = test_ys[i]
				mlp_prob_test_y = model(test_x)
				test_loss = F.binary_cross_entropy(mlp_prob_test_y, test_y)
				mlp_pred_test_y = (mlp_prob_test_y > 0.5) + 0
				mlp_pred_test_y = mlp_pred_test_y.cpu()
				mlp_val_test = evaluation(test_y.cpu(), mlp_pred_test_y.cpu(), mlp_prob_test_y.cpu())  #mlp_thres, mlp_para)
				tmp_val = tmp_val + mlp_val_test
				tmp_loss = tmp_loss + [float(test_loss.data)]
			mlp_vals.append(["l2={},epoch={}".format(l2, epoch)]+tmp_val)
			mlp_loss.append(["l2={},epoch={}".format(l2, epoch)]+tmp_loss)
			model.train()
	model.eval()
	return model, mlp_vals, mlp_loss

def evaluation(y_true, y_pred, y_prob, threshold=None, parameters=None):
	from sklearn import metrics
	f1_score = metrics.f1_score(y_true, y_pred)  
	precision = metrics.precision_score(y_true, y_pred)
	recall = metrics.recall_score(y_true, y_pred)
	balanced_f1_score = metrics.f1_score(y_true, y_pred, average='weighted')
	balanced_precision = metrics.precision_score(y_true, y_pred, average='weighted')
	balanced_recall = metrics.recall_score(y_true, y_pred, average='weighted')
	return [f1_score]


def train_classifiers_simple(config, train_data, test_files):
	test_data = pd.concat(test_files)
	test_length = len(test_files[0])
	print(train_data.shape, test_data.shape)
	train_data = train_data.sample(frac=1)
	data = pd.concat([train_data, test_data], keys=['train', 'test'])
	featured_con_data, gens = feature_encoder(data)

	label_column = config["label_column"]
	train_data = featured_con_data.loc['train']
	test_data = featured_con_data.loc['test']
	X_train = train_data.drop(axis=1, columns=[label_column])
	train_Y = train_data[label_column]
	X_test = test_data.drop(axis=1, columns=[label_column])
	test_Y = test_data[label_column]

	from sklearn import preprocessing
	length = int(len(X_train)/4)
	scaled_train_x = preprocessing.scale(X_train.iloc[:length*3,:])
	scaled_valid_x = preprocessing.scale(X_train.iloc[length*3:,:])
	train_y = train_Y.iloc[:length*3]
	valid_y = train_Y.iloc[length*3:]

	scaled_tests = []
	test_ys = []
	for i in range(len(test_files)):
		scaled_test = preprocessing.scale(X_test.iloc[i*test_length:(i+1)*test_length])
		test_y = test_Y.iloc[i*test_length:(i+1)*test_length]
		scaled_test = torch.from_numpy(scaled_test).float().cuda()
		test_y = torch.from_numpy(test_y.values).float().cuda()
		scaled_tests.append(scaled_test)
		test_ys.append(test_y)
		print(scaled_test.shape, test_y.shape)

	#print(scaled_valid_x.shape, scaled_train_x.shape, scaled_test_x.shape)
	scaled_valid_x = torch.from_numpy(scaled_valid_x).float().cuda()
	scaled_train_x = torch.from_numpy(scaled_train_x).float().cuda()
	train_y = torch.from_numpy(train_y.values).float().cuda()
	valid_y = torch.from_numpy(valid_y.values).float().cuda()

	mlp_vals = []
	mlp_losses = []
	for l2 in config["l2"]:	
		model = MLP(config["input_shape"])
		model.cuda()
		model, mlp_val, mlp_loss = train_simple(model, scaled_train_x, train_y, scaled_tests, test_ys, scaled_valid_x, valid_y, l2, config["epoch"], 0.01)
		mlp_vals = mlp_vals + mlp_val
		mlp_losses = mlp_losses + mlp_loss
	return mlp_vals, mlp_losses

def split_train(train_files):
	train = []
	valid = []
	length = int(len(train_files[0])/4)
	train_keys = []
	for i in range(len(train_files)):
		train_keys.append("train_{}".format(i))
		train.append(train_files[i].iloc[:length*3,:])
		valid.append(train_files[i].iloc[length*3:,:])
	valid_data = pd.concat(valid)
	train_data = pd.concat(train, keys=train_keys)
	return train_keys, train_data, valid_data

def test_concat(test_files):
	for i in range(len(test_files)):
		test_keys.append("test_{}".format(i))
	test_data = pd.concat(test, keys=test_keys)
	return test_keys, test_data	

def train_classifiers_with_consistency(config, train_files, test_files):
	test_data = pd.concat(test_files)
	test_length = len(test_files[0])
	train_data = pd.concat(train_files)
	train_length = len(train_files[0])
	print(train_data.shape, test_data.shape)
	data = pd.concat([train_data, test_data], keys=['train', 'test'])
	featured_con_data, gens = feature_encoder(data)

	label_column = config["label_column"]
	train_data = featured_con_data.loc['train']
	test_data = featured_con_data.loc['test']
	X_train = train_data.drop(axis=1, columns=[label_column])
	train_Y = train_data[label_column]
	X_test = test_data.drop(axis=1, columns=[label_column])
	test_Y = test_data[label_column]

	from sklearn import preprocessing

	scaled_trains = []
	train_ys = []
	scaled_valids = []
	valid_ys = []
	for i in range(len(train_files)):
		X = X_train.iloc[i*train_length:(i+1)*train_length,:]
		Y = train_Y.iloc[i*train_length:(i+1)*train_length]
		length = int(len(X)/4)
		scaled_train_x = preprocessing.scale(X.iloc[:length*3,:])
		scaled_valid_x = preprocessing.scale(X.iloc[length*3:,:])
		train_y = Y.iloc[:length*3]
		valid_y = Y.iloc[length*3:]
		scaled_valids.append(scaled_valid_x)
		valid_ys.append(valid_y)
		scaled_train_x = torch.from_numpy(scaled_train_x).float().cuda()
		train_y = torch.from_numpy(train_y.values).float().cuda()
		scaled_trains.append(scaled_train_x)
		train_ys.append(train_y)
		print(scaled_train_x.shape, train_y.shape)

	scaled_valid_x = np.concatenate(scaled_valids, axis=0)
	valid_y = pd.concat(valid_ys)
	scaled_valid_x = torch.from_numpy(scaled_valid_x).float().cuda()
	valid_y = torch.from_numpy(valid_y.values).float().cuda()
	print(scaled_valid_x.shape, valid_y.shape)

	scaled_tests = []
	test_ys = []
	for i in range(len(test_files)):
		scaled_test = preprocessing.scale(X_test.iloc[i*test_length:(i+1)*test_length])
		test_y = test_Y.iloc[i*test_length:(i+1)*test_length]
		scaled_test = torch.from_numpy(scaled_test).float().cuda()
		test_y = torch.from_numpy(test_y.values).float().cuda()
		scaled_tests.append(scaled_test)
		test_ys.append(test_y)
		print(scaled_test.shape, test_y.shape)

	mlp_vals = []
	mlp_losses = []
	for l2 in config["l2"]:	
		model = MLP(config["input_shape"])
		model.cuda()
		model, mlp_val, mlp_loss = train_consistency(model, scaled_trains, train_ys, scaled_tests, test_ys, scaled_valid_x, valid_y, l2, config["epoch"], 0.01)
		mlp_vals = mlp_vals + mlp_val
		mlp_losses = mlp_losses + mlp_loss
	return mlp_vals, mlp_losses


def generate_augment_data(files):
	file_num = len(files)
	augment = []
	length = int(len(files[0])/len(files))
	for i in range(file_num):
		augment.append(files[i].iloc[i*length:(i+1)*length, :])
	augment_data = pd.concat(augment)
	return augment_data

def thread_run(config):
	logging.basicConfig(filename=config["output"]+'_log.log', level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT)

	train_set = config["train_set"]
	test_set = config["test_set"]

	assert len(train_set) == len(test_set)

	test_files = []
	train_files = []

	for i in range(len(train_set)):
		dtrain = train_set[i]
		dtest = test_set[i]
		logging.info(dtest)
		test_files.append(pd.read_csv(dtest))
		logging.info(dtrain)
		train_files.append(pd.read_csv(dtrain))

	if config["objective"] == "ERM":
		#test_data = generate_augment_data(test_files)
		train_data = pd.concat(train_files)
		mlp_vals, mlp_losses = train_classifiers_simple(config, train_data, test_files)
	elif config["objective"] == "GDRO":
		#test_data = generate_augment_data(test_files)
		mlp_vals, mlp_losses = train_classifiers_with_consistency(config, train_files, test_files)
		
	for i, mlp_val in enumerate(mlp_vals):
		mlp_vals[i] = mlp_val
		columns = ["param", "f1_valid"]
		for i in range(len(test_files)):
			columns.append("f1_test-{}".format(i))
		df = pd.DataFrame(mlp_vals, columns=columns)
		df.to_csv(config["output"]+".csv", index=None)

	for i, mlp_loss in enumerate(mlp_losses):
		mlp_losses[i] = mlp_loss
		columns = ["param", "f1_valid"]
		for i in range(len(test_files)):
			columns.append("f1_test-{}".format(i))
		df = pd.DataFrame(mlp_losses, columns=columns)
		df.to_csv(config["output"]+"_loss.csv", index=None)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('configs', help='a json config file')
	parser.add_argument('gpu', default=0)
	args = parser.parse_args()
	gpu = int(args.gpu)
	torch.manual_seed(0)
	torch.cuda.manual_seed(0)
	torch.cuda.manual_seed_all(0)
	np.random.seed(0)
	if gpu >= 0:
		GPU = True
		os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
	else:
		GPU = False

	with open(args.configs) as f:
		configs = json.load(f)

	jobs = []
	for config in configs:
		thread_run(config)