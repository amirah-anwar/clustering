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
	result_matrix = np.array(result_matrix)
	
	print gmm(result_matrix)

def gmm(result_matrix):
	#Initialize assuming gaussian parameters, get back soft memb
	#Assume/given parameters, mu, cov, and phi, get sof memberships
	#Assume/given soft membership, recalculate parameters
	mu = {}
	temp = random.sample(result_matrix,3)  #get 1 random points represented in numpy arrs
	for k in range(3):
		mu[k] = temp[k]
	D = {0:result_matrix - mu[0], 1:result_matrix - mu[0], 2:result_matrix - mu[0]} #150X2 - vector of 2 elements
	n = float(len(result_matrix))
	cov = {0: (1.0/n)*np.dot(D[0].T, D[0]), 1: (1.0/n)*np.dot(D[1].T, D[1]), 2: (1.0/n)*np.dot(D[2].T, D[2])}
	phi = {0:(1.0/3.0),1:(1.0/3.0), 2:(1.0/3.0)}

	# #For convergence
	# old_factor = math.MAX
	# new_factor = 0.0
	threshold = 200
	while threshold > 0:
		soft_mem = expectation(phi, result_matrix, cov, mu)
		phi, mu, cov = maximization(soft_mem, result_matrix)
		threshold = threshold - 1
	return cov

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

	for k in range(3):
		rik = []
		for i in range(150):
			rik.append(phi[k]*(N(data[i], mu[k], cov[k])/r_denom(phi,data[i],cov, mu)))
		rik = np.array(rik)
		dic[k] = rik
	return dic

def r_denom(phi,data_i, cov, mu):
	sumation = 0.0
	for k in range(3):
		sumation += phi[k]*N(data_i, mu[k], cov[k])
	return sumation

def maximization(soft_mem, data):

	phi = {}
	mu = {}
	cov = {}
	n = float(len(data))

	for k in range(3):
		phi_cluster = 0.0
		mu_cluster = []
		for i in range(150):
			#For each cluster 
			phi_cluster += soft_mem[k][i]
			mu_cluster.append(soft_mem[k][i]*data[i])
		phi[k] = phi_cluster/n
		mu_cluster = np.array(mu_cluster)
		mu[k] = sum(mu_cluster, 0)/phi_cluster
		D = data - mu[k]
		cov[k] = (1.0/n)*np.dot(D.T,D)
	return phi, mu, cov

#Normalization of probability function
def N(x,mu, cov):
	prob = (1/(2*math.pi))*np.power(np.linalg.det(cov), -0.5)*np.exp(np.dot(np.dot(-0.5*((x - mu).T),np.power(cov,-1)),(x - mu)))
	return float(prob)


if __name__ == "__main__":
    main()