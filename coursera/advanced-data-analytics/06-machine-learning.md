# Machine Learning

## Feature Engineering

- Feature Selection
  - Too many features might hurt model performance (e.g. over-fitting)
  - Predictive Features
    - predictive by themselves 
  - Interactive Features
    - predictive in conjunction with other features
  - Irrelevant Features
    - no predictive value
  - highly correlated feature can also be redundant
- Feature Transformation
  - Log Transformation
    - Take log of the skewed feature, to reduce the skew
    - check with histogram
  - Scaling
    - Normalisation
      - Transform data to fall within range [0, 1]
      - ($x_i$ - $x_{mean}$) / ($x_{max}$- $x_{min}$)
    - Standardization
      - Transforms data to have mean of 0 and standard deviation of 1
      - ($x_i$ - $x_{mean}$) / $x_{stand.dev}$
    - Encoding
      - Encode as integer values
- Feature Extraction
https://www.coursera.org/learn/the-nuts-and-bolts-of-machine-learning/supplement/7Q7BT/explore-feature-engineering

## Class Imbalance

- 80/22 imbalance is generally fine, but with 90/10 or more extreme, this needs to be dealt with
- Upsampling
  - with fewer data entries, e.g. a few thousand
- Downsampling
- https://www.coursera.org/learn/the-nuts-and-bolts-of-machine-learning/supplement/qjuKA/more-about-imbalanced-datasets
- Changing the class distribution affects the underlying class probabilities learned by the model

## Naive Bayes

- https://www.coursera.org/learn/the-nuts-and-bolts-of-machine-learning/supplement/1zfDy/naive-bayes-classifiers
- One of the most important is the assumption that each predictor variable (different Bs in the formula) is independent from the others,  conditional on the class. 
- This is called conditional independence.
-  no predictor variable has any more predictive power than any other predictor.0 
- individual predictor variables are assumed to contribute equally to the model’s prediction
- “zero frequency” problem
- https://www.coursera.org/learn/the-nuts-and-bolts-of-machine-learning/supplement/1Uunt/more-about-evaluation-metrics-for-classification-models

## Clustering

- Partitioning vs Clustering
  - Partitioning tries to _partition_ the data space into partitions, therefore no data point will be left out
    - I.e. all points must be assigned to one of the clusters
    - K-Means is a partitioning algorithm
  - Clustering tries to cluster data points in the data space
    - But outlying points could exist outside any of the clusters
- Inter-cluster vs Intra-cluster
  - Inter means "between two things", so "Inter-cluster" is between the given two clusters
    - **Inter-cluster distance** measures the distance between two clusters
  - Intra means "within the given thing"
    - **Intra-cluster distance** measures the distance of data points within a given cluster
  - Higher "Inter-cluster" and Lower "Intra-cluster" are preferred
    - I.e. clusters are separated with greater distance
    - I.e. data points in a given cluster is compacted together
- Inertia
  - Sum of the squared distances between each data point and its nearest centroid
    - $\sum (x - c)^2$
  - Within the same cluster, the distance relationship between data points (observations)
    - Aggregated for all the clusters, as a single score for the metric
  - Mostly about the "intra-cluster" distance
  - Lower inertia is preferred, evaluated by "Elbow Method"
- Silhouette Score
  - Mean of the Silhouette Coefficients of all the observations in the model
    - Coefficients (for each data point): $(b - a) \div \max(a, b)$
      - `a` as the mean distance from "a given data point" to "all other data points" in the same cluster
      - `b` as the mean distance from "a given data point" to "all data points" in the _next closest cluster_
      - value between -1 and 1
        - 1 means a data point is nicely sit within its own cluster and well-separated from closest cluster
        - 0 means a data point is on the boundary of its own cluster
        - -1 means a data point is in the wrong cluster
  - Compared to Inertia, also consider the separation between clusters (i.e. inter-cluster)
  - About both the "inter-" and "intra-" cluster distance
- https://www.coursera.org/learn/the-nuts-and-bolts-of-machine-learning/supplement/V5RFG/more-about-inertia-and-silhouette-coefficient-metrics

### K Means

- Minimize variance between data points in each cluster
- K Means Steps
  1. Randomly place centroids in the data space
  2. Assign each data point to the nearest centroid
  3. Update the location of each centroid
     - As mean position of all the data points assigned to it
  4. Repeat Step 2 and 3 until converges
     - Converged when distance to new position is smaller than a given threshold
- K-means++
  - Randomly initialises centroids based on a probability calibration
  - Choosing a data point as the first centroid
    - Next use other data points as centroids
    - Probability of being a data point selected as a centroid increases the farther it is from other centroids
    - Ensure that initial centroids are not placed together (thus less likely to converge in local minima)
- Using distance between observations means you have to "scale" the data before fitting
- https://www.coursera.org/learn/the-nuts-and-bolts-of-machine-learning/supplement/GUuy9/more-about-k-means

### Other Algorithms

- Why? Different way to cluster data in the data space
- DBSCAN
  - Density-based Spatial Clustering of Applications with Noise
  - Searches for high-density region in data space
- Agglomerative clustering
  - Start by assign every point to its own cluster
    - Then combine the cluster by inter-cluster distance
- https://www.coursera.org/learn/the-nuts-and-bolts-of-machine-learning/supplement/yynh1/clustering-beyond-k-means

## Decision-Trees

- Pros
  - No assumptions regarding the data distribution
  - Handles collinearity (unlike logistic or linear regression)
- Cons
  - Prone to over-fitting
- Impurity 
  - Split on the criterion that minimize the impurity
  - Degree of mixture with respect to class
  - Impurity = 0 means the child nodes contain only N for one class and 0 for another class
  - High Impurity is when both child nodes have equal number for both classes
  - ...
- Gini Impurity
  - ...
- https://www.coursera.org/learn/the-nuts-and-bolts-of-machine-learning/supplement/zShQK/explore-decision-trees
