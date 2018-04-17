"""
A usage example for bhatta_dist() using the well-known iris data set.

Created on 4/14/2018
Author: Eric Williamson (ericpaulwill@gmail.com)
"""

import pandas
import numpy as np
from bhatta_dist import bhatta_dist

###Download the iris data set and separate the feature matrix from the class array.
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width']
dataset = pandas.read_csv(url, names=names+['class'])
array = dataset.values
n_features = array.shape[1]-1
X = array[:,0:-1]
Y = array[:,-1:].reshape(-1)

###For this example we will just look at the Bhattacharyya distance between the 'versicolor' class and the 'setosa' class.
Y_SELECTION = ['Iris-setosa', 'Iris-versicolor']
bh_dists = np.zeros(n_features)
for i,name in enumerate(names):
    #Take the i-th feature and divide it by class:
    X1 = np.array(X[:,i],dtype=np.float64)[Y==Y_SELECTION[0]]
    X2 = np.array(X[:,i],dtype=np.float64)[Y==Y_SELECTION[1]]
    #Call the Bhattacharyya distance function (we'll just use the default method)
    bh_dists[i] = bhatta_dist(X1, X2)

#Print the result for each feature:
for name,d in zip(names,bh_dists):
    print("Bhattacharyya distance of {}: {:.2f}".format(name, d))
#Use the results to rank features:
print("Feature ranking:")
print(np.array(names)[bh_dists.argsort()[::-1]])
