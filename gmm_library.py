import itertools

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn import mixture

color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold',
                              'darkorange'])


def plot_results(X, Y_, means, covariances, index, title):
    splot = plt.subplot(2, 1, 1 + index)
    for i, (mean, covar, color) in enumerate(zip(
            means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)
    for i in range(3):
		lines = plt.plot(means[i,0],means[i,1],'kx')
		plt.setp(lines,ms=15.0)
		plt.setp(lines,mew=2.0)


    # plt.xlim(-9., 5.)
    # plt.ylim(-3., 6.)
    # plt.xticks(())
    # plt.yticks(())
    plt.title(title)

f = open("clusters.txt", 'r')
result_matrix = []
for line in f.readlines():
	values_as_strings = line.split(',')
	arr = np.array(map(float, values_as_strings))
	result_matrix.append(arr)
data = np.array(result_matrix)

gmm = mixture.GaussianMixture(n_components=3, covariance_type='full', init_params="random").fit(data)
print "mus", gmm.means_
plot_results(data, gmm.predict(data), gmm.means_, gmm.covariances_, 0,
             'Gaussian Mixture')
plt.show()