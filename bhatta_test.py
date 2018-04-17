"""
Use Gaussian distributions to randomly generate two sets. Then use bhatta_dist() on the sets. Compare the results to the
 theoretical Bhattacharyya distance for the distributions.

The Bhattacharyya distance between two Gaussian distributions is given on this page:
  https://en.wikipedia.org/wiki/Bhattacharyya_distance

Created on 4/17/2018
Author: Eric Williamson (ericpaulwill@gmail.com)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from bhatta_dist import bhatta_dist


###We will generate our distributions from these parameters:
sigma1 = 0.5 #standard deviation
mu1 = 2.0    #mean
n1 = 5000    #population size
sigma2 = 1.5
mu2 = 3.1
n2 = 10000

###Calculate theoretical Bhattacharyya distance:
var1 = sigma1**2
var2 = sigma2**2
bdist_theory = np.log( (var1/var2 + var2/var1 + 2)/4 ) / 4  +  (mu1-mu2)**2 / (var1+var2) / 4

###Generate random data that follows our normal distributions:
X1 = np.random.normal(mu1,sigma1,n1)
X2 = np.random.normal(mu2,sigma2,n2)

###Plot distributions of X1 and X2 to verify that we generated the data we want:
def get_density(x,cov_factor=0.1):
    #Produces a continuous density function for the data in 'x'. Some benefit may be gained from adjusting the cov_factor.
    #Note that cov_factor is not related to scaling or bias.
    density = gaussian_kde(x)
    density.covariance_factor = lambda:cov_factor
    density._compute_covariance()
    return density
N_STEPS = 200
d1 = get_density(X1)
d2 = get_density(X2)
xs = np.linspace(0,10,N_STEPS)
fig, ax = plt.subplots()
p1 = np.exp(-((xs-mu1)**2) / (2*var1)) / np.sqrt(2*np.pi*var1)
p2 = np.exp(-((xs-mu2)**2) / (2*var2)) / np.sqrt(2*np.pi*var2)
ax.plot(xs,d1(xs))
ax.plot(xs,d2(xs))
ax.plot(xs,p1)
ax.plot(xs,p2)
ax.legend(['X1 density','X2 density','theoretical 1','theoretical 2'])
plt.show()

###Use bhatta_dist() function to calculate Bhattacharyya distance:
#Note that the 'noiseless' method is not tested here, since the feature we've generated is quantitative.
bdist_cont = bhatta_dist(X1,X2,method='continuous')
bdist_hist = bhatta_dist(X1,X2,method='hist')
bdist_ahist = bhatta_dist(X1,X2,method='autohist')
#Show results:
print("Theoretical:       {:.3f}".format(bdist_theory))
print("Test 'continuous': {:.3f}   Error: {:.6f}".format(bdist_cont, bdist_cont-bdist_theory))
print("Test 'hist':       {:.3f}   Error: {:.6f}".format(bdist_hist, bdist_hist-bdist_theory))
print("Test 'autohist':   {:.3f}   Error: {:.6f}".format(bdist_ahist, bdist_ahist-bdist_theory))

###Test 'noiseless' method by binning the values first (makes no sense for a real dataset, just for testing):
cX = np.concatenate((X1,X2))
N_BINS = 20
h1 = np.histogram(X1,bins=N_BINS,range=(min(cX),max(cX)), density=False)[0]
h2 = np.histogram(X2,bins=N_BINS,range=(min(cX),max(cX)), density=False)[0]
fakeX1 = []
fakeX2 = []
for i in range(N_BINS):
    fakeX1 += [i] * h1[i]
    fakeX2 += [i] * h2[i]
bdist_nl = bhatta_dist(fakeX1,fakeX2,method='noiseless')
print("Test 'noiseless':  {:.3f}   Error: {:.6f}".format(bdist_nl, bdist_nl-bdist_theory))
