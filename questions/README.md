# Questions

List of questions you should be able to answer with ease

## List of Questions

### General

- What are the differences between supervised and unsupervised learning?
  - Labels or not, i.e. build-in feedback mechanism
  - E.g. Decision Tree, Logistic Regression and SVM
  - E.g. K-Means clustering, Hierarchical clustering, Apriori algorithm
- How to avoid over-fitting your model?
  - Keep the model simpler, e.g. use fewer features, check the correlation between features and discard highly correlated ones
  - Use cross-validation method, e.g. k-fold cross validation
  - TODO: Use regularization techniques
- What is the difference between univariate, bivariate, and multivariate analysis?
  - Univariate: describing the only variable
  - Bivariate: investigating the relationship between two variables, e.g. position index and click rate of links in a newsletter
  - Multivariate: involving three or more variables, but similar technique with bivariate
- TODO: What are the feature selection methods used to select the right variables?
  - Filter Methods, filter out the features by performing tests
    - Linear Discrimination Analysis
    - ANOVA
    - Chi-Square
  - Wrapper Methods: incremental or decremental set of features
    - Forward Selection: 1 -> 2 -> 3 ... -> all
    - Backward Selection: all -> ... -> 3 -> 2 -> 1
    - Recursive Feature Elimination: different pairs of features
- How to deal with missing values?
  - If larger dataset, simply removing the records of missing values
  - If smaller dataset, replace with mean or median
- What are dimensionality reduction and its benefits?
  - Transforming dataset with N feature to M features, where N > M. But try to keep as much information as possible
  - Useful to reduce computation, storage space, removing redundant features
  - Extract meaningful features from data, e.g. Topic Modelling
- TODO: How will you calculate eigenvalues and eigenvectors of the following 3x3 matrix?
- What are recommender systems?:
  - Collaborative Filtering
  - Content-based Filtering
  - Hybrid approach
- How can outlier values be treated?
  - Try plotting the distribution, either scatterplot or histogram
  - If outlier value is impossible, remove it
  - If possible, try scale the data with square root, natural log ...etc
  - Or, use models less affected by outlier
- TODO: How can time-series data be declared as stationery?
- TODO: How to calculate Entropy with a given list of 0 and 1?
- What is the difference between the long format data and wide format data?
- What is variance?
  - Used to understand the distribution of data, deviation from the mean
  - 
- Do gradient descent methods always converge to similar points?
- What is the law of large numbers?
- What are the confounding variables?
- What is star schema?
- What are eigenvalue and eigenvector?
- What is selection bias?
- What is a bias-variance trade-off?
- Describe Markov chains?
- Difference between an error and a residual error
- Difference between Normalisation and Standardization
- Difference between Point Estimates and Confidence Interval
- Explain the most challenging data science project that you worked on.
- What does root cause analysis mean?
- What is A/B testing?
- Explain TF/IDF vectorization.

### Statistics

- What are some techniques for sampling?
  - Probability-based
    - Simple Random Sampling
    - Stratified Sampling
    - 


### Machine Learning algorithms

- What are the assumptions required for linear regression?
  - Linear Relationship: there should be a linear relationship between independent variable x and dependent variable y
  - Independence: there is no correlation between consecutive residuals. It mostly occurs in time-series data.
  - Homoscedasticity: at every level of x, the variance should be constant
  - Normality: the residuals are normally distributed
- What happens when some of the assumptions required for linear regression are violated?
- How is logistic regression done?
  - Relationship between dependent and independent variable
  - Sigmoid function
  -
- Explain the steps in making a decision tree
- How to build a Random Forest
  - Built with a given number of decision trees
- What is pruning in a decision tree algorithm?
  - Pruning simplifies the decision tree by reducing the rules
  - Helps by reducing complexity
- What is entropy in a decision tree algorithm?
  - Entropy is the measure of randomness or disorder in the group of observations
  - Entropy is also used to check the homogeneity of the given data
    - If the entropy is zero, then the sample of data is entirely homogeneous
    - If the entropy is one, then it indicates that the sample is equally divided
- What is "Information Gain" in a decision tree algorithm?
  - Information gain is the expected reduction in entropy
- What is a kernel function in SVM?
- What is ensemble learning?
- Explain bagging, boosting and stacking in Data Science
- What does the word ‘Naive’ mean in Naive Bayes?


- Explain collaborative filtering in recommender systems.
- Explain content-based filtering in recommender systems.


### Deep Learning

- What is an RNN (recurrent neural network)?
- What is Gradient Descent?
  - An iterative first-order optimization process called gradient descent (GD) is used to locate the local minimum and maximum of a given function. This technique is frequently applied in machine learning (ML) and deep learning (DL) to minimize a cost/loss function (for example, in a linear regression).
- What is Dropout?
  - In Data Science, the term “dropout” refers to the process of randomly removing visible and hidden network units. By eliminating up to 20% of the nodes, they avoid overfitting the data and allow for the necessary space to be set up for the network’s iterative convergence process.
- What is batch normalization?
- What is the Computational Graph?
- What is the difference between Batch and Stochastic Gradient Descent?
-  What is an Activation function?

### Metrics

- TODO: How do you find RMSE and MSE in a linear regression model?
- How can you select k for k-means?
  - Elbow method: plot the residuals by varying K, finding K with turning point in the plot
- TODO: What is the significance of p-value?
- How can you calculate accuracy using a confusion matrix?
  - (True Positive + True Negative) divided by number of observations
- Write the equation and calculate the precision and recall rate
  - Precision: (True Positive) / (True Positive + False Positive)
  - Recall: (True Positive) / (True Positive + False Negative)
  - TODO: interpretation
- What do you understand about true positive rate and false-positive rate?
  - The True Positive Rate (TPR) defines the probability that an actual positive will turn out to be positive
  - The False Positive Rate (FPR) defines the probability that an actual negative result will be shown as a positive one i.e the probability that a model will generate a false alarm
- What is the ROC curve? 
  - Used to evaluate binary classification models
  - Plot of TPR and FPR with multiple threshold values
  - If it's a diagonal line, it means the model has the same performance as randomly guessing
- What is the F1 score and how to calculate it?
- Define the terms KPI, lift, model fitting, robustness, and DOE.

### Deployment

- How should you maintain a deployed model?
  - Monitor, Evaluate, Compare and Rebuild

### Database

- How is Data modeling different from Database design?
  - Data Modelling:
    - Creates a conceptual model based on the relationship between various data models
    - Its process involves moving from the conceptual stage to the logical model to the physical schema
  - Database Design:
    - Creates an output which is a detailed data model of the database
    - Includes the detailed logical model of a database but it can also include physical design choices and storage parameters

## Resources

- [Top 100+ Data Science Interview Questions in 2023](https://intellipaat.com/blog/interview-question/data-science-interview-questions/?US)
- [21 Top Data Scientist Interview Questions](https://www.datacamp.com/blog/data-scientist-interview-questions)
