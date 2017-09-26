# ==============Group Members==================================
# Michelle Becerra
# Amirah Anwar
# Reetinder Kaur

import random
import numpy as np
from numpy import sum
import sys
import math

def main():
	#load txt file into an array of numpy arrays
	f = open("clusters.txt", 'r')
	result_matrix = []
	cov = {}
	for line in f.readlines():
	    values_as_strings = line.split(',')
	    arr = np.array(map(float, values_as_strings))
	    result_matrix.append(arr)

	#Initialize
	mu = np.array(random.sample(result_matrix, 3))
	#print "initial mu:",mu
	N = float(len(result_matrix))
	for k in range(3):
		D = result_matrix - mu[k]
		cov[k] = (1/N)*np.dot(D.T,D)
	phi = [(1.0/3.0),(1.0/3.0), (1.0/3.0)]
	old_factor = sys.float_info.max
	c_factor = 0.0
	while old_factor > c_factor:
		old_factor = c_factor
		soft_mem = expectation(phi, np.array(result_matrix), cov, mu)
		#print "soft membership:",soft_mem
		phi, mu, cov = maximization(soft_mem,np.array(result_matrix))
		c_factor = convergence(phi, mu, cov, result_matrix)
		#print "old_factor:",old_factor
		#print "c_factor:",c_factor
	print "mu:",mu

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
			#a = N(data[i],mu[k],cov[k])
			rik.append(phi[k]*N(data[i], mu[k], cov[k])/r_denom(phi,data[i],mu, cov))
		print " values:",phi[k],mu[k],cov[k],rik,k
		dic[k] = rik
	return dic

def r_denom(phi,data, mu, cov):
	sumation = 0.0
	for i in range(3):
		sumation += phi[i]*N(data, mu[i], cov[i])
	return sumation

def maximization(soft_mem, data):
	#mu = mu(result_matrix);
	phi = {}
	mu = {}
	phi_cluster = 0.0
	temp_data = []
	mu_cluster = np.array([])
	N = float(len(data))
	cov = {}
	for k in range(3):
		for i in range(150):
			phi_cluster += soft_mem[k][i]
			temp_data.append(soft_mem[k][i]*data[i])
		phi[k] = phi_cluster/N
		mu[k] = sum(temp_data, 0)/phi_cluster
		#print "mu[k]:",mu[k],k
		D = data - mu[k]
		cov[k] = (1/N)*np.dot(D.T,D)
	return phi, mu, cov

def N(x,mu, cov):
	
	#print "values for i",x,mu,cov
	return (1/(2*math.pi))* math.pow(np.linalg.det(cov), -0.5) * math.exp(-0.5*np.dot(np.dot((x - mu).T,np.reciprocal(cov)),(x - mu)))

# def mu(data):
# 	arr = np.array(np.mean(data, axis=0))
# 	return arr

if __name__ == "__main__":
    main()
