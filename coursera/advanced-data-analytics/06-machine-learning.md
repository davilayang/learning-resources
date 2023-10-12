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

## K Means

- https://www.coursera.org/learn/the-nuts-and-bolts-of-machine-learning/supplement/GUuy9/more-about-k-means
