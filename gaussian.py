# ==============Group Members==================================
# Michelle Becerra mdbecerr@usc.edu
# Amirah Anwar anwara@usc.edu
# Reetinder Kaur reetindk@usc.edu

import itertools
import random
import numpy as np
from numpy import sum
import math
from numpy.linalg import inv
import clustering
from matplotlib import pyplot
from scipy import linalg
import matplotlib as mpl
from matplotlib import gridspec

color_iter = itertools.cycle(['navy', 'turquoise', 'red'])
def main():
	#load txt file into an array of numpy arrays
	f = open("clusters.txt", 'r')
	result_matrix = []
	for line in f.readlines():
	    values_as_strings = line.split(',')
	    arr = np.array(map(float, values_as_strings))
	    result_matrix.append(arr)
	result_matrix = np.array(result_matrix)

	#number of clusters
	k=3

	#gaussian mixture model parameters calculation with kmeans initialization
	method = "k-means"
	mu_kmean, phi_kmean, cov_kmean, predictions = gmm(result_matrix, method, k)

	#creating three subplots
	fig = pyplot.figure()
	gs = gridspec.GridSpec(3, 3)

	print "GMM with k-means initialization"
	output_print(mu_kmean, cov_kmean, phi_kmean)

	ax1 = fig.add_subplot(gs[0,:])
	plot_results(result_matrix, predictions, mu_kmean, cov_kmean, 0, 'Gaussian Mixture with k-means initialization', ax1, fig, gs)

	print "----------------------------------------------------"

	#gaussian mixture model parameters calculation with random initialization
	method = "random"
	mu_random, phi_random, cov_random, predictions = gmm(result_matrix, method, k)

	print "GMM with random initialization"
	output_print(mu_random, cov_random, phi_random)

	ax2 = fig.add_subplot(gs[1,:])
	plot_results(result_matrix, predictions, mu_random, cov_random, 1, 'Gaussian Mixture with random initialization', ax2, fig, gs)

	print "----------------------------------------------------"	

	#clustering of datapoints with kmeans algorithm
	centroids, clusters = clustering.kmeans(result_matrix, k)
	trueCentroid = clustering.trueCentroids(centroids, result_matrix, k)

	print "Calculated centroids by K-Means Algorithm"
	for centroid in trueCentroid:
		print centroid
	plot_kmeans(clusters, trueCentroid, fig, gs)
	fig.tight_layout()
	pyplot.show()

def output_print(means, covariances, phis):
	print "Computed means of each cluster:"
	for index in means:
		print means[index]
	print "Computed amplitudes of each cluster:"
	for index in phis:
		print phis[index]
	print "Computed covariance of each cluster:"
	for index in covariances:
		print covariances[index]

def plot_kmeans(clusters, trueCentroid, fig, gs):
	ax3 = fig.add_subplot(gs[2,:])
	i = 0
	for key, value in clusters.iteritems():
		value =  np.array(value)
		#plot of all datapoints
		ax3.scatter(value[:,0],value[:,1], marker='o')
		#plot of true centroids of clusters
		lines = ax3.plot(trueCentroid[i][0],trueCentroid[i][1],'kx')
		pyplot.title("K-Means")
		i += 1
	#axis set up
	pyplot.axis([-5,10,-6,10])
	ax3.grid(which='both')
	pyplot.setp(ax3.get_xticklabels(), fontsize=8)
	pyplot.setp(ax3.get_yticklabels(), fontsize=8)

def plot_results(X, Y_, means_dic, covariances_dic, index, title, ax, fig, gs):
	#adding subplot for gaussian ellipses
	splot = fig.add_subplot(gs[index,:])

	means = []
	covariances = []
	for i in range(len(means_dic)):
		means.append(means_dic[i])
	for i in range(len(covariances_dic)):
		covariances.append(covariances_dic[i])

	#axis set up
	ax.grid(which='both')
	pyplot.axis([-5,10,-6,10])
	pyplot.setp(ax.get_xticklabels(), fontsize=8)
	pyplot.setp(ax.get_yticklabels(), fontsize=8)

	#plot of datapoints
	ax.scatter(X[:,0], X[:,1], marker='o')

	#plotting gaussian ellipses 
	for i, (mean, covar, color) in enumerate(zip(means, covariances, color_iter)):
		v, w = linalg.eigh(covar)
		v = 2. * np.sqrt(2.) * np.sqrt(v)
		u = w[0] / linalg.norm(w[0])

        # Plot an ellipse to show the Gaussian component
		angle = np.arctan(u[1] / u[0])
		angle = 180. * angle / np.pi  # convert to degrees
		ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
		ell.set_clip_box(splot.bbox)
		ell.set_alpha(0.2)
		#ellipse plot command
		splot.add_artist(ell)

	means = np.array(means)
	#plotting mus
	ax.scatter(means[:,0], means[:,1], marker='x')

	pyplot.title(title)

def gmm(result_matrix, method, k):
	#Assume/given parameters: mu, cov, and phi, get sof memberships
	#Assume/given soft membership, recalculate parameters
	mu = {}
	if method == "random":
		temp = random.sample(result_matrix,k)  #get 3 random points represented in numpy arrs

	if method == "k-means":
		centroids, clusters = clustering.kmeans(result_matrix, k)
		temp = clustering.trueCentroids(centroids, result_matrix, k)

	for k in range(3):
		mu[k] = temp[k]

	n = float(len(result_matrix))
	# sample_mean = sample_mean/n
	D = {0:result_matrix - mu[0], 1:result_matrix - mu[1], 2:result_matrix - mu[2]} #150X2 - vector of 2 elements

	cov = {0: (1.0/n)*np.dot(D[0].T, D[0]), 1: (1.0/n)*np.dot(D[1].T, D[1]), 2: (1.0/n)*np.dot(D[2].T, D[2])}
	phi = {0:(1.0/3.0),1:(1.0/3.0), 2:(1.0/3.0)}

	#For convergence
	old_factor = 10000
	new_factor = convergence(phi, mu, cov, result_matrix)

	#EM algorithm for calculating gaussian parameters
	soft_mem =	None
	while (abs(old_factor) - abs(new_factor)) > 0.01:
		old_factor = new_factor
		#Expectation step
		soft_mem = expectation(phi, result_matrix, cov, mu)
		#Maximization step
		phi, mu, cov = maximization(soft_mem, result_matrix)
		#Convergence factor
		new_factor = convergence(phi, mu, cov, result_matrix)

	predictions = predict(soft_mem)
	return mu, phi, cov, predictions

#Cluster prediction for datapoints
#Choosing the cluster for a datapoint which has the highest probability
#of having that datapoint
def predict(soft_mem):
	predictions = np.array([])
	temp = []
	for i in range(150):
		maxi = None
		for k in range(3):
			temp_max = soft_mem[k][i]
			if maxi < temp_max:
				maxi = temp_max
				key = k
		temp.append(key)
	predictions = np.array(temp)
	return predictions

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
			temp_rik.append((phi[k]*N(data[i], mu[k], cov[k]))/r_denom(phi,data[i],cov, mu))
		rik = np.array(temp_rik)
		dic[k] = rik
	return dic

def r_denom(phi,data_i, cov, mu):
	sumation = 0.0
	for k in range(3):
		sumation += (phi[k]*N(data_i, mu[k], cov[k]))
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