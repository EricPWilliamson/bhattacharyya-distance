# bhattacharyya-distance
Computes the Bhattacharyya distance for feature selection in machine learning.


The function accepts discrete data and is not limited to a particular probability distribution (eg. a normal Gaussian distribution). Included are four different methods of calculating the Bhattacharyya coefficient--in most cases I recommend using the 'continuous' method.

In it's current form, the function can only accept one feature at at time, and can only compare two classes. Use multiple function calls to analyze multiple features and multiple classes.

It is not necessary to apply any scaling or normalization to your data before using this function.
