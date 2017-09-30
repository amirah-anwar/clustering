import numpy as np
from sklearn import cluster
from matplotlib import pyplot

#Using scilearn-kit to plot and get the values of centroids for our data

f = open("clusters.txt", 'r')
result_matrix = []
for line in f.readlines():
	values_as_strings = line.split(',')
	arr = np.array(map(float, values_as_strings))
	result_matrix.append(arr)
data = np.array(result_matrix)

kmeans = cluster.KMeans(n_clusters=3)
kmeans.fit(data)

labels = kmeans.labels_
centroids = kmeans.cluster_centers_
print centroids

#Plot the data
for i in range(3):
	ds = data[np.where(labels==i)]
	pyplot.plot(ds[:,0],ds[:,1],'o')
	lines = pyplot.plot(centroids[i,0],centroids[i,1],'kx')
	pyplot.setp(lines,ms=15.0)
	pyplot.setp(lines,mew=2.0)
	pyplot.title("K-Means")
pyplot.show()