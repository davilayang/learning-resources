# About Nearest Neighbour

<!-- markdownlint-disable MD033 -->

<style>
p { text-indent: 5%; }
li { margin-left: -15px; }
color1 { color: crimson; }
</style>

## Intuitions of Nearest Neighbour

[k-nearest neighbors algorithm](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)

KNN is a type of **instance-based learning, or lazy learning**, where the function is only approximated locally and _all computation is deferred until classification_. The KNN algorithm is among the simplest of all machine learning algorithms.  

Both for classification and regression, a useful technique can be used to **assign weight to the contributions of the neighbors**, so that the nearer neighbors contribute more to the average than the more distant ones. For example, _a common weighting scheme consists in giving each neighbour a weight of `1/d`, where `d` is the distance to the neighbour_.

A peculiarity of the KNN algorithm is that it is <color1>sensitive to the local structure of the data</color1>

### Distance Metrics

A commonly used distance metric for continuous variables is **Euclidean distance**. For discrete variables, such as for text classification, another metric can be used, such as the overlap metric (or **Hamming distance**).  

In the context of gene expression microarray data, for example, KNN has also been employed with correlation coefficients such as Pearson and Spearman. Often, the classification accuracy of KNN can be improved significantly if the distance metric is learned with specialized algorithms such as _Large Margin Nearest Neighbour or Neighbourhood components analysis_.  

## Steps of K Nearest Neighbour Algorithm

1. Load data as training and testing set
2. Initialise the value of `k`
   - I.e. the number of nearest neighbours to consider
3. For each data point in testing set:
   - Calculate its distance to all the points in training, using distance metrics like "Euclidean", "Hamming"...etc
   - Sort the training points by distance
   - Take the top `K` points and get their labels
   - Use the majority label as the test point's label

## Backgrounds of Nearest Neighbour

Euclidean distance

- Square root of the squared differences between two data points

Hamming distance

- Measures the minimum number of substitutions required to change one string into the other

## Appendix: References

- [k-nearest neighbors algorithm](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)
- [KNN Numerical Example](https://people.revoledu.com/kardi/tutorial/KNN/KNN_Numerical-example.html)
- [A Complete Guide to K-Nearest Neighbors (Updated 2023)](https://www.analyticsvidhya.com/blog/2018/03/introduction-k-neighbours-algorithm-clustering/)
