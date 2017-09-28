# ==============Group Members==================================
# Michelle Becerra
# Amirah Anwar
# Reetinder Kaur

import random
import numpy as np
import sys
#Classifies a data set using K-means Algo,
#Reads the data file
#Number of clusters = 3 = k
#Keeps on calculating the centroids of the data set until centroids converge
def main():
    # ==============Process Data==================================

    # #load txt file into an array of numpy arrays
	f = open("clusters.txt", 'r')
	result_matrix = []
	for line in f.readlines():
	    values_as_strings = line.split(',')
	    arr = np.array(map(float, values_as_strings))
	    result_matrix.append(arr)

	#Pre-defined number of clusters
	k = 3

	#K-mean algorithm for classification
	#First run
	centroids, clusters = kmeans(result_matrix, k)
	trueCentroid = trueCentroids(centroids, result_matrix, k)

	print "Calculated centroids"
	for centroid in trueCentroid:
		print centroid

#Selects centroids randomly
#1.Assigns each data point to it's nearest centroid and forms a cluster
#2.Re-calculates the centroid for each cluster
#Repeats steps 1 and 2 until centroids converge
def kmeans(dataSet, k):
	centroids = random.sample(dataSet, k)
	# print "randomly selected centroids:", centroids

	prevCentroids = None
	#Runs until centroids converge
	while not np.array_equal(prevCentroids, centroids):
		prevCentroids = centroids
		#Assign each data point to it's nearest centroid and forms a cluster
		clusters = assignment(dataSet, centroids)
		#Re-calculates the centroid for each cluster
		centroids = reComputation(clusters)
		# print "re-computed centroids:", centroids
	return centroids, clusters

#Re-calculates the centroid for each cluster
#by calculating the mean of each cluster
def reComputation(clusters):
	newCentroids = []
	for cluster in clusters:
		arr = np.array(np.mean(clusters[cluster], axis=0))
		newCentroids.append(arr)
	return newCentroids

#Assigns each data point to it's nearest centroid and forms a cluster
def assignment(dataSet, centroids):
	clusters = {}
	for dataPoint in dataSet:
		selectedCentroid = closest(dataPoint, centroids)
		selectedCluster = str(selectedCentroid)
		if selectedCluster not in clusters:
			clusters[selectedCluster] = [dataPoint]
		else:
			clusters[selectedCluster].append(dataPoint)
	return clusters

#Finds the closest centroid to a data point
def closest(dataPoint, centroids):
    return min(centroids, key = lambda centroid: distance(dataPoint, centroid))

#Calculates Euclidean distance of a centroid and a data point
def distance(x, y):
	return np.sqrt(sum((x - y) ** 2))

def squaredeuclidean(arr):
#Take first set of points and get its square
	summ = 0.0
	ab = sum((arr[0] - arr[1])**2)
	summ += sum((ab - arr[2])**2)
	return summ
    #summ += numpy.sum((ab - arr[2])**2)

def trueCentroids(centroids, result_matrix, k):
	#Optimization after scilearn-kit
	#Run the kmeans algo 9 more times and get the lowest total squared Euclidean Distance
	minOfRun = squaredeuclidean(centroids)
	trueCentroid = centroids
	for i in range(9):
		currCentroid, clusters = kmeans(result_matrix,k)
		currMin = squaredeuclidean(currCentroid)
		if(currMin < minOfRun):
			minOfRun = currMin
			trueCentroid = currCentroid
	return trueCentroid

if __name__ == "__main__":
    main()
