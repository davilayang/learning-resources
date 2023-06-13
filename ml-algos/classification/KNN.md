# About Nearest Neighbor

## Intuitions of Nearest Neighbor

[k-nearest neighbors algorithm](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)
k-NN is a type of **instance-based learning, or lazy learning**, where the function is only approximated locally and _all computation is deferred until classification_. The k-NN algorithm is among the simplest of all machine learning algorithms.  
Both for classification and regression, a useful technique can be used to **assign weight to the contributions of the neighbors**, so that the nearer neighbors contribute more to the average than the more distant ones. For example, _a common weighting scheme consists in giving each neighbor a weight of 1/d, where d is the distance to the neighbor_.

A peculiarity of the k-NN algorithm is that it is sensitive to the local structure of the data

### Distance Metrics

A commonly used distance metric for continuous variables is Euclidean distance. For discrete variables, such as for text classification, another metric can be used, such as the overlap metric (or Hamming distance). In the context of gene expression microarray data, for example, k-NN has also been employed with correlation coefficients such as Pearson and Spearman.[3] Often, the classification accuracy of k-NN can be improved significantly if the distance metric is learned with specialized algorithms such as Large Margin Nearest Neighbor or Neighbourhood components analysis. 

## Steps of k Nearest Neighbor Algorithm

## Backgrounds of Nearest Neighbor

Euclidean distance

Hamming distance

## Appendix: References
