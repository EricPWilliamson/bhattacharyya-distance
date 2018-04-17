# bhattacharyya-distance
Computes the Bhattacharyya distance for feature selection in machine learning.


The function accepts discrete data and is not limited to a particular probability distribution (eg. a normal Gaussian distribution). Included are four different methods of calculating the Bhattacharyya coefficient--in most cases I recommend using the 'continuous' method.

CONTENTS:
bhatta_dist.py  -Contains functions for calculating Bhattacharyya distance.

iris_example.py -Usage example

test_bhatta.py  -Verification of the calculations in bhatta_dist()

In it's current form, the function can only accept one feature at at time, and can only compare two classes. Use multiple function calls to analyze multiple features and multiple classes.

It is not necessary to apply any scaling or normalization to your data before using this function. However, other forms of preprocessing that might alter the class separation within the feature should be applied prior.
