# ==============Group Members==================================
# Michelle Becerra
# Amirah Anwar
# Reetinder Kaur

import random
import numpy as np
from numpy import sum
import math
from numpy.linalg import inv

def main():
	#load txt file into an array of numpy arrays
	f = open("clusters.txt", 'r')
	result_matrix = []
	for line in f.readlines():
	    values_as_strings = line.split(',')
	    arr = np.array(map(float, values_as_strings))
	    result_matrix.append(arr)
	result_matrix = np.array(result_matrix)
	calculated_mu = gmm(result_matrix)
	print "re_computed mu", calculated_mu

def gmm(result_matrix):
	#Initialize assuming gaussian parameters, get back soft memb
	#Assume/given parameters, mu, cov, and phi, get sof memberships
	#Assume/given soft membership, recalculate parameters
	mu = {}
	temp = random.sample(result_matrix,3)  #get 1 random points represented in numpy arrs
	print "random centroids", temp
	for k in range(3):
		mu[k] = temp[k]
	print "random mu", mu
	D = {0:result_matrix - mu[0], 1:result_matrix - mu[1], 2:result_matrix - mu[2]} #150X2 - vector of 2 elements
	# print "D", D
	n = float(len(result_matrix))
	cov = {0: (1.0/n)*np.dot(D[0].T, D[0]), 1: (1.0/n)*np.dot(D[1].T, D[1]), 2: (1.0/n)*np.dot(D[2].T, D[2])}
	phi = {0:(1.0/3.0),1:(1.0/3.0), 2:(1.0/3.0)}
	# print "n", n
	# print "cov", cov
	# print "phi", phi
	#For convergence
	old_factor = None
	new_factor = convergence(phi, mu, cov, result_matrix)
	# threshold = 200
	n = 0
	list_factor = []
	list_factor.append(new_factor)

	while old_factor < new_factor:
		old_factor = new_factor
		soft_mem = expectation(phi, result_matrix, cov, mu)
		phi, mu, cov = maximization(soft_mem, result_matrix)
		# summation = 0.0
		# for k in range(3):
		# 	summation += soft_mem[k][0]
		# print "sum of first rik", summation
		# print "phi, mu, cov", phi, mu, cov
		new_factor = convergence(phi, mu, cov, result_matrix)
		list_factor.append(new_factor)
		n += 1
		# threshold = threshold - 1

	print "iterations", n
	print "convergence factor", list_factor
	return mu

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
		temp_rik = []
		for i in range(150):
			temp_rik.append(phi[k]*(N(data[i], mu[k], cov[k])/r_denom(phi,data[i],cov, mu)))
		rik = np.array(temp_rik)
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
		tmp_mu_cluster = []

		for i in range(150):
			#For each cluster 
			# print "soft_mem[k][i]", soft_mem[k][i]
			phi_cluster += soft_mem[k][i]
			tmp_mu_cluster.append(soft_mem[k][i]*data[i])
		phi[k] = phi_cluster/n
		mu_cluster = np.array(tmp_mu_cluster)
		mu[k] = sum(mu_cluster, 0)/phi_cluster
		D = data - mu[k]
		cov[k] = (1.0/n)*np.dot(D.T,D)
	return phi, mu, cov

#Normalization of probability function
def N(x,mu, cov):
	prob = (1/(2*math.pi))*np.power(np.linalg.det(cov), -0.5)*np.exp(np.dot(np.dot(-0.5*((x - mu).T),inv(cov)),(x - mu)))
	return float(prob)


if __name__ == "__main__":
    main()