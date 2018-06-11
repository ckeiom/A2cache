import numpy as np
import math
import sys
from numpy.linalg import inv
from scipy.stats import multivariate_normal

np.set_printoptions(threshold = 20000,precision = 2, suppress = 1);
num_bucket = 100;
# Train
data_x = np.loadtxt('x.txt', dtype = int, delimiter=',');
data_x = data_x.reshape(len(data_x), num_bucket, 2);
data_y = np.loadtxt('y.txt', dtype = int);
valid_pos = 0;
distance = np.zeros(num_bucket);
for i in range(num_bucket, len(data_x)):
	index = (data_x[i] == 0);
	if(index.any()):
		continue;
	else:
		valid_pos = i + 1;
		break;

train_x = np.zeros((len(data_x) - valid_pos, 2), dtype = float);
train_y = np.zeros((len(train_x)), dtype = int);
f_value = np.zeros((num_bucket), dtype = int)

for i in range(len(train_x)):
	for j in range(num_bucket):
		f_value[j] = data_x[valid_pos + i, j, 1];
		if(f_value[j] > 100):
			f_value[j] = 100;
	train_x[i, 0] = np.mean(f_value);
	train_x[i, 1] = np.var(f_value);
	train_y[i] = data_x[valid_pos + i, data_y[valid_pos + i], 0];
for i in range(len(train_y)):
	print(train_x[i], train_y[i]);
pi = np.zeros((num_bucket), dtype = float);
mu = np.zeros((num_bucket, 2), dtype = float);
sigma = np.zeros((num_bucket, 2, 2), dtype = float);
epsilon = 0.001;

for i in range(num_bucket):
	indices = (train_y == (i + 1));
	if(indices.any()):
		pi[i] = float(sum(indices)) / float(len(train_y));
		mu[i] = np.mean(train_x[indices, :], axis = 0);
		sigma[i] = np.cov(train_x[indices, :], rowvar = 0, bias = 1);
		sigma[i, 0, 0] += epsilon;
		sigma[i, 1, 1] += epsilon;
	else:
		pi[i] = 0;
		mu[i][0] = 0;
		mu[i][1] = 0;
		sigma[i] = np.zeros((2, 2), dtype = float);

factor = np.zeros((2), dtype = float);
factor[0] = 4;
factor[1] = 10;

for label in range(num_bucket):
	if(mu[label][0] != 0):
		rv = multivariate_normal(mean = mu[label], cov = sigma[label]);
		distance[label] = rv.logpdf(factor);
	else:
		distance[label] = -9999;
	
indices = (distance != 0);
r_value = np.argmax(distance);


#print(mu);
#print(sigma);
#print(distance);
#print(r_value);
"""
for i in range(valid_pos, len(data_y)):
	print("a"+str(data_x[i, data_y[i], 0])+"a");
"""

#print(train_x);
#print(train_x);
#print(sigma);
# Test
