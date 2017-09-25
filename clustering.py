# ==============Group Members==================================
# Michelle Becerra
# Amirah Anwar
# Reetinder Kaur

import random
import numpy as np

def main():
    # ==============Process Data==================================

    # #load txt file into a ndarray
	f = open("clusters.txt", 'r')
	result_matrix = []
	for line in f.readlines():
	    values_as_strings = line.split(',')
	    arr = np.array(map(float, values_as_strings))
	    result_matrix.append(arr)

	print "read file:", result_matrix

	k = 3

	centroids = kmeans(result_matrix, k)
	# print "Calculated centroids", centroids


def kmeans(dataSet, k):
	centroids = random.sample(dataSet, k)
	print "centroids:", centroids

	prevCentroids = None
	# while not (prevCentroids == centroids)
	# prevCentroids = centroids
	clusters = assignment(dataSet, centroids)
	print "clusters", clusters
	# centroids = computation(dataSet, k)
	return centroids

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

def closest(dataPoint, centroids):
    return min(centroids, key = lambda centroid: distance(dataPoint, centroid))

def distance(x, y):
	return np.sqrt(sum((x - y) ** 2))

if __name__ == "__main__":
    main()
