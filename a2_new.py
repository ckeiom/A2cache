import numpy as np
import math
import sys
from numpy.linalg import inv

np.set_printoptions(threshold=np.inf);
# Train
data_x = np.loadtxt('x.txt', dtype = int, delimiter=',');
data_y = np.loadtxt('y.txt', dtype = int);
valid_pos = 0;
num_bucket = 100;

for i in range(len(data_x[0]), len(data_x)):
	index = (data_x[i] == 0);
	if(index.any()):
		continue;
	else:
		valid_pos = i;
		break;

train_x = np.zeros((len(data_x) - valid_pos, 3), dtype = float);

for i in range(len(train_x)):
	train_x[i, 0] = data_x[valid_pos + i, data_y[valid_pos + i] * 2];
	train_x[i, 1] = data_x[valid_pos + i, data_y[valid_pos + i] * 2 + 1];
	f_sum = 0;
	for j in range(num_bucket):
		f_val = data_x[valid_pos + i, j * 2 + 1];
		f_sum += f_val;
	train_x[i, 2] = f_sum / num_bucket; 

mu = np.mean(train_x, axis = 0);
sigma = np.cov(train_x, rowvar = 0, bias = 1);
epsilon = 0.001;

if(sigma[0, 0] == 0):
	sigma[0, 0] = epsilon;
if(sigma[1, 1] == 0):
	sigma[1, 1] = epsilon;
if(sigma[2, 2] == 0):
	sigma[2, 2] = epsilon;
print(valid_pos);
#print(train_x);
print(mu);
print(sigma);
#print(train_x);
#print(sigma);
# Test

num_bucket = 100;
crude_data = np.loadtxt(sys.argv[1], dtype = int);
candidate = np.zeros((len(crude_data)), dtype = int);
output = np.zeros((len(crude_data), num_bucket, 2), dtype = int);
cache = [0] * num_bucket;
num_eviction = 0;
distance = [len(crude_data)] * num_bucket;


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
	f_sum = 0;
	for k in range(num_bucket):
		f_sum += output[i, k, 1];
	factor = np.zeros(3);

	# Update distances (in this case, possibility)
	for j in range(num_bucket):
		factor[0] = output[i, j, 0];
		factor[1] = output[i, j, 1];
		factor[2] = f_sum / num_bucket;
		constant = 1 / (pow(2 * math.pi, 2) * pow(np.linalg.det(sigma), 1/2));
		exp = -1/2 * np.dot(inv(sigma), (factor - mu));
		exp = np.dot((factor - mu).transpose(), exp);
		distance[j] = constant * np.exp(exp);

	# Choose a new candidate
	if(any(x == 0 for x in cache)):
		candidate[i] = cache.index(0);
	else:
		candidate[i] = np.argmax(distance);
print("\nA2_new");
print("Miss: " + str(num_eviction));
print("Cold Miss: " + str(num_bucket));
np.savetxt('trainx.txt', train_x, fmt = '%d', delimiter = ',');
