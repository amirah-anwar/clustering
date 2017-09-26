# ==============Group Members==================================
# Michelle Becerra
# Amirah Anwar
# Reetinder Kaur

import random
import numpy as np
from numpy import sum
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
	
def em(result_matrix):
	#Initialize
	mu = ramdom.sample(result_matrix, 3)
	D = result_matrix - mu
	N = len(result_matrix)
	cov = (1/N)*D.T*D
	phi = [(1.0/3.0),(1.0/3.0), (1.0/3.0)]
	c_factor = math.MAX
	while old_factor != c_factor:
		old_factor = c_factor
		soft_mem = expectation(phi, result_matrix, cov, mu)
		phi, mu, cov = maximization(soft_mem)
		c_factor = convergence(phi, mu, cov, result_matrix)

def convergence(phi, mu, cov, data):
	c_factor = 0.0
	for i in range(150):
		sumation = 0.0
		for k in range(3):
			sumation += phi[k]*N(data[i], mu[k], cov[k])
		c_factor += math.log(sumation)
	return c_factor

def expectation(phi, data, cov, mu):
	dic = {}
	rik = []
	for k in range(3):
		for i in range(150):
			rik.append(phi[k]*N(data, mu, cov)/r_denom(phi,data,mu, cov))
		dic[k] = rik
	return dic

def r_denom(phi,data, mu, cov):
	sumation = 0.0
	for i in range(3):
		sumation += phi[i]*N(data, mu, cov)
	return sumation

def maximization(soft_mem, data):
	#mu = mu(result_matrix);
	phi = {}
	mu = {}
	phi_cluster = 0.0
	temp_data = []
	mu_cluster = np.array([])
	N = len(data)
	cov = {}
	for k in range(3):
		for i in range(150):
			phi_cluster += soft_mem[k][i]
			temp_data.append(soft_mem[k][i]*data[i])
		phi[k] = phi_cluster/N
		mu[k] = sum(temp_data, 0)/phi_cluster
		D = data - mu[k]
		cov[k] = (1/N)*D.T*D
	return phi, mu, cov

def N(x,mu, cov):
	1/(2*math.pi)*math.pow(np.linalg.det(cov), -0.5)*math.exp(-0.5*(x - mu).T*math.pow(cov,-1)*(x - mu))

# def mu(data):
# 	arr = np.array(np.mean(data, axis=0))
# 	return arr

if __name__ == "__main__":
    main()