import tensorflow as tf
import numpy as np
import sys

num_bucket = 100;
crude_data = np.loadtxt(sys.argv[1], dtype = int);
output = np.zeros((len(crude_data), num_bucket, 2), dtype = int);
candidate = np.zeros((len(crude_data)), dtype = int);
cache = [0] * num_bucket;
num_eviction = 0;
distance = [len(crude_data)] * num_bucket;

for i in range(len(crude_data)):
#print(candidate[i]);
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
	# Update distances
	for j in range(num_bucket):
		distance[j] = num_bucket - output[i, j, 0];
	
	# Choose a new candidate
	if(any(x == 0 for x in cache)):
		candidate[i] = cache.index(0);
	else:
		candidate[i] = np.argmax(distance);

print("\nLRU");
print("Miss: " + str(num_eviction));
print("Cold Miss: " + str(num_bucket));
