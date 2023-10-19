# Machine Learning Algorithms

> Understand the ML algorithms

## Sections

- TODO: for each algorithm, add a section on "assumption"

### Basic Concepts

- Supervised Learning vs. Unsupervised Learning

### Supervised Algorithms

- Regression
  - [ ] Linear Regression
- Classification
  - [ ] Logistic Regression
  - [ ] Support Vector Machine (SVM)
  - [ ] Decision Tree
  - [ ] Random Forest
  - [ ] Naive Bayes
  - [ ] XGBoost
- Neural Network
  - [ ] ...
- ...

### UnSupervised Algorithms

- Dimensionality Reduction
  - [ ] Principal Component Analysis (PCA)
  - [ ] NMF
  - ...
- Clustering
  - [ ] K Means
  - [ ] K-Nearest Neighbour
  - [ ] ...
...

### Metrics & Evaluations

- Regression Evaluations
  - [ ] Mean Absolute Error
    - _Error as the difference between target and predicted value_
    - Take average on the absolute value of error
  - [ ] Median Absolute Error
    - Take median on the absolute value of error
  - [ ] Mean Squared Error, MSE
    - Take average on the squared value of error
  - [ ] Mean Squared Log Error
    - Take average on the squared and natural log value of error
  - [ ] Root Mean Squared Error
    - Take average on the squared and natural log value of error
  - [ ] R Squared, R² Coefficient of determination
    - _“How much (what %) of the total variation in Y(target) is explained by the variation in X(regression line)”_
    - Take 1, minus the percentage of variation described by the regression line
    - I.e. the percentage NOT described by the regression line
  - [ ] Adjusted R Squared

- Classification Evaluations
  - [ ] Accuracy
  - [ ] Confusion Matrix
    - True Positive
    - True Negative
    - False Positive: _Falsely classified as Positive (i.e. label is Negative)_
      - Type I Error
    - False Negative: _Falsely classifier as Negative (i.e. label is Positive)_
      - Type II Error
  - [ ] Precision and Recall
  - [ ] F1-score
  - [ ] AUC-ROC

## Questions you should be able to answer

- When to use a metric to evaluate a model performance?
- What is a Lasso Regression? What is a Ridge Regression?
- Describe gradient descent to a child

## References

- [Performance Metrics in Machine Learning \[Complete Guide\]](https://neptune.ai/blog/performance-metrics-in-machine-learning-complete-guide)
