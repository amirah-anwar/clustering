# ==============Group Members==================================
# Michelle Becerra
# Amirah Anwar
# Reetinder Kaur

import random
import numpy as np
import math

def main():
	#load txt file into an array of numpy arrays
	f = open("clusters.txt", 'r')
	result_matrix = []
	for line in f.readlines():
	    values_as_strings = line.split(',')
	    arr = np.array(map(float, values_as_strings))
	    result_matrix.append(arr)

	em(result_matrix)
def em():
	#Initialize
	mu = ramdom.sample(result_matrix, 3)
	D = result_matrix.T - mu.T
	cov = (1/N)*D*D.T
	phi = [(1.0/3.0),(1.0/3.0), (1.0/3.0)]

def expectation():


def maximization(result_matrix):
	#mu = mu(result_matrix);

	N = len(result_matrix)
	
	
	for k in range(3):
		for i in range(150):
			r_ic = phi[k]*N(result_matrix[i], mu[k], cov)
def N(x,mu, cov):
	1/(2*math.pi)*math.pow(np.linalg.det(cov), -0.5)*math.exp(-0.5*(x - mu).T*math.pow(cov,-1)*(x - mu))
def mu(data):
	arr = np.array(np.mean(data, axis=0))
	return arr

if __name__ == "__main__":
    main()