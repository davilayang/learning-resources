# Regression

## PACE with Regression

1. Plan
   - Understanding your data with the problem context
   - Business Know-how
2. Analyse
   - Choosing one or more models might be appropriate
   - Performing EDA
   - Checking model assumptions
     - Statements about the data that must be true to justify the use of particular data techniques
3. Construct
   - Building the model
     - Selecting variables and transforming data
   - Rechecking model assumptions after model built
4. Execute
   - Evaluating the model
     - Choosing metrics and comparing models
   - Iterative refine the models (PACE)

## Model Assumptions

### Linear Regression

- https://www.coursera.org/learn/regression-analysis-simplify-complex-data-relationships/supplement/FZ28t/the-four-main-assumptions-of-simple-linear-regression
- **Linearity**
  - Linear relationship between the variables
  - Check by scatter plot with Variable A and Variable B
    - `sns.pairplot` to plot multiple pair between multiple variables
  - NOTE: If violated, try
    - Take the logarithm to transform data (or other transformations)
- **Independent Observations**
  - Observations are independent to each other
  - Check data collection process
  - NOTE: If violated, try
    - Take just a subset of the available data
- **Normality**
  - Errors (Residuals) are normally distributed
    - I.e. difference between observed and predicted for each data point is normally distributed
    - Errors: the _natural noises_ assumed to be in the model
    - Residuals: the _difference_ between predicted and observed values
      - Calculated after building the model
  - Check by Histogram plot of residuals
  - Check by Q-Q plot between the residuals and Normal Distribution
    - After model is constructed, get their residuals `model.resid`
    - Then, import the modules `import statsmodels.api as sm`
    - Then, plot Quartile-Quartile plot by `sm.qqplot(model.resid, line='s')`
  - NOTE: If violated, try
    - Take the logarithm on dependent variable (y) or X
- **Homoscedasticity**
  - Residuals have constant or similar variance across the model
    - I.e. the difference between observed and predicted is within a similar range
  - Check by residual plots
    - I.e. scatter plot between residuals and fitted values (i.e. predicted values)
  - NOTE: If violated, try
    - Define a different dependent variable
    - Transform the Y variable

### Multiple Linear Regressions

- **Linearity**
  - Each independent variable (X) is linearly related to the dependent variable (y)
    - Independent = Predictor, Dependent = Outcome
  - Check by:
    - Scatter plot between each X and y, expect somewhat linear relationship between them
      - E.g. `sns.pairplot`
- **Independent observations**
  - Each observation in the dataset is independent to each other
  - Check by:
    - Review the data collection process
- **(Multivariate) normality**
  - Errors (Residuals) are normally distributed
    - I.e. the difference between predicted and observed values are normally distributed
  - Check by: (AFTER creating the model)
    - Q-Q Plot, between Model Residuals and Normal Distribution
      - Expect plot shows a diagonal line of around 45 degrees
    - Histogram of the Residuals, if is close to Normal DistributionN
      - Expect to be like Normal Distribution
- **Homoscedasticity**
  - Variation of the errors is constant or similar across the model
  - Check by:
    - Residual Plot, i.e. scatter plot between residuals and fitted values (i.e. predicted values)
      - Expect the variation on Y-axis within a constant/similar range across whole X-axis
- **No multicollinearity**
  - Any of the two independent variables (X) are NOT highly correlated with each other
  - Check by
    - Scatter plot between X variables
      - Expect NO straight line
    - Variance Inflation Factors (VIF)
      - Quantify how much the variance of each variable is "inflated" due to correlation with other X variables
      - `from statsmodels.stats.outliers_influence import variance_inflation_factor`
      - VIF = 1 => No Correlation
  - NOTE: If violated, try
    - Keep only one of the correlated variables
      - Forward Selection
      - Backward Elimination
    - Ridge Regression
    - Lasso Regression
    - Principal component analysis (PCA)
- https://www.coursera.org/learn/regression-analysis-simplify-complex-data-relationships/supplement/8DHXf/multiple-linear-regression-assumptions-and-multicollinearity

### Variable Selections

- Forward Selection
  - Begins with null model (0 independent variable)
- Backward Elimination
  - Begins with full model (all of the independent variables)
- Extra-Sum-of-Squares F-test
  - ...
- Regularization
  - Lasso Regression
    - Completely remove the X variables that are not important in predicting y
    - NOTE: Lasso AS "套索", so non-relevant variables are looped and removed from the equation!
  - Ridge Regression
    - Minimize the impact from X variables that are not important
    - Not removing any X
  - Elastic Net Regression

    - Test both above, or the hybrid of the two


## Model Interpretations

> Results from `statsmodels.api`

- Linear Regression Results
  - Coefficients `p-values` (for each coefficient)
    - Hypotheses
      - Null: coefficient = 0
      - Alternative: coefficient != 0
    - If `p-value` < significance level, reject the null hypothesis
      - The coefficient is statistically significant
      - The given X (independent) is correlated to y (dependent)
  - Coefficients `Confidence Intervals`
    - With 95% Confidence Interval
      - the interval has a 95% chance of containing the true parameter value of the coefficient
      - if you were to repeat this experiment many times, 95% of the confidence intervals would contain the true value
    - Confidence Band
      - the line that describes the uncertainty around the predicted outcome

## Model Evaluations

- https://www.coursera.org/learn/regression-analysis-simplify-complex-data-relationships/supplement/y4kkN/evaluation-metrics-for-simple-linear-regression
- **R-Squared**
  - Coefficient of determination, value between 0 and 1; the higher the better
  - **Proportion of variation in y that is explained by X**
  - To get better model explanations: proportion of variation that is explained
  - Formula
    - `1 - Unexplained Variation`
    - Unexplained Variation = 
      - `(Sum of Squared Residuals)`, divided by `Total Sum of Squares`
  - Issues
    - With more X variables, difficult to tell which contributes to the explained variations
    - Higher R-Squared might happen due to over-fitting
- **Adjusted R-Squared**
  - Between 0 and 1
  - R-Squared but **penalises unnecessary explanatory variable**
  - To compare models of varying complexity: adding a new variable or not
- **MAE**, Mean Absolute Error
  - Average of the absolute difference between the predicted and actual values
  - Good to use with data having outliers that should be excluded/ignored in evaluation
  - Not too sensitive to large errors
- **MSE**, Mean Squared Error
  - Average of the squared difference between the predicted and actual values
  - Sensitive to large errors (because "squared")
- **Bias & Variance trade-off**
  - High Bias: under-fit with the sample data
    - Over-simplify the variable relationships by making assumptions
    - NOTE: Bias-Under
    - NOTE: Simply saying `y = 2` is a "High Bias (under-fitting)" model
  - High Variance: over-fit with the sample data
    - Learned from existing data and incorporate flexibility and complexity
    - NOTE: Variance-Over
  - https://www.coursera.org/learn/regression-analysis-simplify-complex-data-relationships/supplement/jtFae/underfitting-and-overfitting

## Snippets

**We really want to get down to which pages or which actions that people are taking on a website that's indicative of them being a high-value customer or someone who makes multiple purchases within a year.**

Causation describes a cause and effect relationship where one variable directly causes the other to change in a particular way.
Proving causation statistically requires much more rigorous methods and data collection than correlation.
For a data professional, the distinction between correlation and causation is especially important when presenting results.
Articulating that correlation is not causation is part of a data professionals best practices and ethical toolbox.

These estimated Betas are also called regression coefficients.
Now, whenever you see the hat symbol, you'll know that you are estimating Betas, also known as regression coefficients.
(Population parameters do not use hat)

Question: What's an independent variable?

Answer: An independent variable is exactly what it sounds like. It is a variable that stands alone and isn't changed by the other variables you are trying to measure. For example, someone's age might be an independent variable. Other factors (such as what they eat, how much they go to school, how much television they watch) aren't going to change a person's age. In fact, when you are looking for some kind of relationship between variables you are trying to see if the independent variable causes some kind of change in the other variables, or dependent variables.

Question: What's a dependent variable?

Answer: Just like an independent variable, a dependent variable is exactly what it sounds like. It is something that depends on other factors. For example, a test score could be a dependent variable because it could change depending on several factors such as how much you studied, how much sleep you got the night before you took the test, or even how hungry you were when you took it. Usually when you are looking for a relationship between two things you are trying to find out what makes the dependent variable change the way it does.

Many people have trouble remembering which is the independent variable and which is the dependent variable. An easy way to remember is to insert the names of the two variables you are using in this sentence in they way that makes the most sense. Then you can figure out which is the independent variable and which is the dependent variable:

(Independent variable) causes a change in (Dependent Variable) and it isn't possible that (Dependent Variable) could cause a change in (Independent Variable).

##
