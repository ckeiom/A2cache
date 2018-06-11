import numpy as np
import math
import sys
import random
from numpy.linalg import inv
from scipy.stats import multivariate_normal

np.set_printoptions(threshold=np.inf);
num_bucket = 100;
num_label = 4;
group_size = num_bucket / num_label;
# Train
data_x = np.loadtxt('x.txt', dtype = int, delimiter=',');
data_x = data_x.reshape(len(data_x), num_bucket, 2);
data_y = np.loadtxt('y.txt', dtype = int);
valid_pos = 0;

for i in range(len(data_x[0]), len(data_x)):
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
		if(f_value[j] > num_bucket / 2):
			f_value[j] = num_bucket / 2;
	train_x[i, 0] = np.mean(f_value);
	train_x[i, 1] = np.var(f_value);
	train_y[i] = data_x[valid_pos + i, data_y[valid_pos + i], 0];

print(train_y);
#print(valid_pos);
pi = np.zeros((num_label), dtype = float);
mu = np.zeros((num_label, 2), dtype = float);
sigma = np.zeros((num_label, 2, 2), dtype = float);
epsilon = 0.001;

for i in range(num_label):
	indices = ((i * group_size < train_y) * \
			   (train_y <= ((i + 1) * group_size)))
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
print(mu);
print(pi);
print(sigma);
#print(train_x);
#print(mu);
#print(sigma);
#print(train_x);
#print(sigma);
# Test

crude_data = np.loadtxt(sys.argv[1], dtype = int);
candidate = np.zeros((len(crude_data)), dtype = int);
output = np.zeros((len(crude_data), num_bucket, 2), dtype = int);
cache = [0] * num_bucket;
num_eviction = 0;
distance = [len(crude_data)] * num_label;


for i in range(len(crude_data)):

	# Copy from previous state
	if(i > 0):
		output[i] = output[i - 1];
		candidate[i] = candidate[i - 1];
	

	index = (cache == crude_data[i]);

	# Cache Hit
	if(index.any()):
		LRU_before = output[i, index, 0];
		output[i, index, 0] = num_bucket;
		output[i, index, 1] += 1;
		for j in range(num_bucket):
			if((cache[j] != crude_data[i]) and \
					(output[i, j, 0] > LRU_before)):
				output[i, j, 0] -= 1;

	# Cache Miss
	else:
		num_eviction += 1;
		cache[candidate[i]] = crude_data[i];
		LRU_before = output[i, candidate[i], 0];
		output[i, candidate[i], 0] = num_bucket;
		output[i, candidate[i], 1] = 1;
		for j in range(num_bucket):
			if((cache[j] != crude_data[i]) and \
					(output[i, j, 0] > LRU_before)):
				output[i, j, 0] -= 1;
	
	factor = np.zeros((2));
	for j in range(num_bucket):
		f_value[j] = output[i, j, 1];
		if(f_value[j] > 100):
			f_value[j] = 100;
	
	factor[0] = np.mean(f_value);
	factor[1] = np.var(f_value);
	# Update distances (in this case, possibility)
	for label in range(num_label):
		if(mu[label][0] != 0):
			rv = multivariate_normal(mean = mu[label], cov = sigma[label]);
			distance[label] = np.log(pi[label]) +rv.logpdf(factor);
		else:
			distance[label] = -99999;
	"""
	for j in range(num_bucket):
#print(j);
#		print(mu[j]);
#		print(sigma[j]);
		if(mu[j][0] != 0):
			distance[j] = rv.logpdf(
			constant =  1 / (pow(2 * math.pi, 2) * pow(np.linalg.det(sigma[j]), 1/2));
			exp = -1/2 * np.dot(inv(sigma[j]), (factor - mu[j]));
			exp = np.dot((factor - mu[j]).transpose(), exp);
			distance[j] = constant * np.exp(exp);
		else:
			distance[j] = 0;
			"""

	# Choose a new candidate
	if(any(x == 0 for x in cache)):
		candidate[i] = cache.index(0);
	else:
		label = np.argmax(distance);
		r_value = (label + 1) * group_size #random.randrange(1, group_size);
		for j in range(num_bucket):
			if(output[i, j, 0] == r_value):
				candidate[i] = j;
#	print(distance);
		print(i, factor, r_value, label);

print("\nA2_OPT");
print("Miss: " + str(num_eviction));
print("Cold Miss: " + str(num_bucket));
np.savetxt('candi.txt', candidate, fmt = '%d');
np.savetxt('mu.txt', mu, fmt = '%f', delimiter = ',');
