# About Logistic Regression

<!-- markdownlint-disable MD033 -->

<style>
p { text-indent: 5%; }
li { margin-left: -15px; }
color1 { color: crimson; }
</style>

## Intuitions of Logistic Regression

With linear regression, the output is always a number that has its real meaning. Theoretically, the number can be within the range of $-\infty$ and $+\infty$.  

However, the output for a logistic regression is **a number that represents the probability of the event happening** (e.g. the probability of people clicking an ad online, the probability of death in titanic disaster, etc.).  

The intuition behind logistic regression is to <color1>transform the output of a linear regression, which has a wider range, to a range that probability lies in, which is within $[0,1]$. The transformation formula is using **Logit function** that maps a value to a number in the range of $[0,1]$</color1>.  

1. Underlying logistic regression is a latent (unobservable) linear regression model:
    + $y^* = X\beta + u$
      + $y^*$ is a continuous unobservable variable
      + $X$ is the regressor matrix
      + $\beta$ is the paramter vector
    + error term is assumed, which follows the logistic distribution, $u\mid X\sim \Lambda(0, \frac {\pi^2}{3})$
2. Assume that the observed binary variable $y$ is an Indicator function of the unobservable $y^*$
    + $y = 1 \;\; \text{if} \;\; y^* > 0$
    + $y = 0 \;\; \text{if} \;\; y^* \le 0$
3. The question is "what is the probability that $y$ will take the value $1$ given the regressors"
    + we are looking at a conditional probability
    + $P(y=1 \mid X) = P(y^*>0 \mid X) = P(X\beta + u>0 \mid X) = P(u> - X\beta\mid X)$
    + $\;\; = 1- \Lambda (-Χ\beta) = \Lambda (X\beta)$ (due to symmetry property of logistic distribution)
4. The basic logistic regression model
    + probability of event happending, $p$
    + $p=P(y=1 \mid X)  = \Lambda (X\beta) = \cfrac{1}{1+e^{-X\beta}}$

In regression analysis, logistic regression (or logit regression) is estimating the parameters of a logistic model; it is a form of **binomial** regression. In the logistic model, <color1>the **log-odds (the logarithm of the odds) for the value labeled "1" is a linear combination (hence a linear model) of one or more independent variables (the predictors)**</color1>; the independent variables can each be a binary variable (two classes, coded by an indicator variable) or a continuous variable (any real value).  

 The corresponding _probability_ of the value labeled "1" can vary between 0 (certainly the value "0") and 1 (certainly the value "1"), hence the labeling; **the function that converts log-odds to probability is the logistic function, hence the name**. The unit of measurement for the log-odds scale is called a _logit_, from _logistic unit_, hence the alternative names.

## Steps of Logistic Regression Algorithm

<!-- just algorithm steps, add some instructions as the comments in codes -->
<!-- later update with regularizations -->

1. Prepare Dataset and Variables initializations
    + continuous explanatory variables $(x)$ w/ categorical response variable $(y=[0, 1])$
    + add bias term to dataset
    + initilize weights for training
2. Compute logits, or log-odds by matrix multiplication
    + by multipling feature vectors with weights
    + underlying is a latent (unobservable) linear regression model
    + logit transformation
      + $ \beta X =  \beta_0 + \beta_1X_1 + \beta_2X_2 + ... + \beta_kX_k = \log \big(\frac{P}{1-P} \big) = logit(P)$
3. Reconstrcut probability from logits/log-odds
    + using logistic function, also called sigmoid function
      + $\frac{e^{x\beta}}{1 + e^{x\beta}} = Probability$
4. Compute gradients with Maximum Likelihood Estimation
    + using log-likelihood or average log-likelihood
5. Update the weights
6. Optimization goal is to
    + **minimize loss/cost function**
    + **maximize (average) log-likelihood**

## Backgrounds of Logistic Regression

1. Link function
2. Logit function
3. Logistic function
4. Logit and Logistic function in Logistic Regression
5. Odds and example
6. Odds-ratio and example
7. Maximum Likelihood Estimation, MLE

### Link function

[Link Function, wikipedia](https://en.wikipedia.org/wiki/Generalized_linear_model#Link_function)  

A link function provides the relationship between **the linear predictor** and **the mean of the distribution function**. There are many commonly used link functions, and their choice is informed by several considerations. There is always a well-defined canonical link function which is derived from the exponential of _the response's density function_, e.g. compare response of linear regression with generalized linear model:  

+ Linear Regression
  + assumes that the response variable is normally distributed
    + with response ranging from $-\infty$ to $+\infty$
+ Generalized Linear Model (just some examples)
  + can be of Bernoulli distribution, with response $\{0, 1\}$
  + can be of Binomial distribution, with response $\{0, 1, ..., N\}$
  + can be of Exponential, Poisson, Categorical, Multinomial ... distribution

 Link function links **the mean of the dependent variable $Y$**, i.e. the expected value $E(Y)=μ$ to **the linear term (or linear combination) $\boldsymbol{X\beta}$** in such a way that the range of the non-linearly transformed mean $g(μ)$ ranges from $-\infty$ to $+\infty$. Thus you can actually form a linear equation $g(μ) = X\beta$ and use an iteratively reweighted least squares method for maximum likelihood estimation of the model parameters. The link function **transforms the probabilities of the levels of a categorical response variable (e.g. $\{0, 1\}$) to a continuous scale that is unbounded ($-\infty, +\infty$)**. Once the transformation is complete, the relationship between the predictors and the response can be modeled with linear regression.  

+ $\gamma(\theta)$ is the cumulant moment generating function
+ $g(\mu)$ is the link function
+ link function $g$ relates the linear predictor to the mean
+ link function provides the relationship between
  + the linear predictor and
    + linear function (linear combination) of a set of coefficients and explanatory
    + i.e. $X\beta$
  + the mean of the distribution function
+ the diagram allows to easily go from one direction to the other
  + $\eta = g \left( \gamma(\theta)\right)$
  + $\theta = \gamma'^{-1}\left( g^{-1}(\eta)\right)$

<!-- When $Y$ is categorical, we use the _logit of $Y$_ as the response in our regression equation instead of just $Y$ itself:  
$\ln{\bigg(\cfrac{P}{1-P}\bigg)} = \beta_0 + \beta_1X_1 + ... + \beta_kX_k$, where $P$ is defined as the probability that $Y=1$  

The logit function is the natural log of the odds that Y equals one of the categories.  For mathematical simplicity, we’re going to assume Y has only two categories and code them as 0 and 1. -->

### Logit function

[Logit, wikipedia](https://en.wikipedia.org/wiki/Logit)  

In statistics, the logit function or the log-odds is the logarithm of the odds $\frac{p}{1 − p}$ where $p$ is the probability. It is a type of function that **creates a map of probability values from $[0,1]$ to $[-\infty ,+\infty]$**. It is the _inverse of the sigmoidal "logistic" function_ or logistic transform used in statistics.  

Logit, the unit of measurement for the log-odds scale, is shortened from _Logistic Unit_. A **logit is defined as the log base $e$ (log) of the odds**, i.e. $logit(p) = \log(odds) = \log\big(\frac{p}{1-p}\big)$. The range of logit is from **negative infinity to positive infinity**.  

+ odds have range from $0$ to $infinity$
+ $\log(1)$ is $0$, $\log(0)$ is $-\infty$, and $\log(\infty)$ is $+\infty$
+ **therefore, $logit(p) = \log(odds)$ ranges from $-\infty$ to $+\infty$**

If we call the parameter $\theta$, it is defined as follows: $logit(\theta) = \log(\frac{\theta}{1 - \theta})$. The **logistic** is a _the inverse of the logit_. if we have a value $x$, the logistic is: $logistic(x) = \frac{e^x}{1 + e^x}$.

### Logistic function

[Logistic function, wikipedia](https://en.wikipedia.org/wiki/Logistic_function)  

Logistic function, a part of sigmoid function family, can handle a large set of continuous independent variables ($X$) and produce a binary output. Its form is: $h_\theta = \frac{1}{1 + e^{-z}} = \frac{1}{1 + e^{-\theta \cdot x}}$. The logistic function is the inverse of the natural logit function and so can be used to convert the logarithm of odds into a probability.

Due to its sigmoid-shape and bounded in y-direction by 0 and 1, it's widely used to handle classification problems. The **decision boundary** of logistic regression is usually _chosen at the middle of the logistic function, namely at $z=0$ where the output value $y$ is $0.5$_.  
$$
y = \left\{
        \begin{array}{ll}
            1, \text{if } z > 0 \\
            0, \text{if } z < 0
        \end{array}
    \right.
$$

### Sigmoid functions family

[Sigmoid function](https://en.wikipedia.org/wiki/Sigmoid_function)  

A sigmoid function is a mathematical function having a characteristic _"S"-shaped curve_ or sigmoid curve. _**Often, sigmoid function refers to the special case of the logistic function**. Sigmoid functions have domain of all real numbers, with_ return value monotonically increasing most often from $0$ to $1$ or alternatively from $−1$ to $1$_, depending on convention.  

Some of the sigmoid function family:  

+ Logistic Function, $f(x) = \frac{1}{1 + e^{-x}}$
+ Hyperbolic tangent (shifted and scaled version of the logistic function), $f(x) = \tanh x = \frac{e^x - e^{-x}}{e^x + e^{-x}}$

### Inverse relationship between Logit and Logistic

<!--  We know that logistic is the inverse of logit, and the we want to estimate the parameter $\theta$ to achieve a model to classify the independent variable $y$.  Here, we'll be using matrix notation where $X$ is an $N \times p$ matrix and $\beta$ is a $p \times 1$ vector. -->

+ **logit function**: $logit(\theta) = \log\big(\frac{\theta}{1-\theta}\big) = X\beta$
+ **logistic function**: $\theta = \frac{e^{X\beta}}{1+e^{X\beta}}$
+ Inverse relatonship:
  + $\log\big(\frac{\theta}{1-\theta}\big) = X\beta$
  + $\iff e^{\log(\frac{\theta}{1-\theta})} = e^{X\beta} \iff \frac{\theta}{1-\theta} = e^{X\beta} $
  + $\iff \theta = e^{X\beta} - \theta \cdot e^{X\beta} \iff \theta(1 + e^{X\beta}) = e^{X\beta}$
  + $\iff \theta = \frac{e^{X\beta}}{1+e^{X\beta}}$

### Odds

[Odds, wikipedia](https://en.wikipedia.org/wiki/Odds)  

Odds are determined from probabilities _ranging between **0** and **infinity**_ and defined as _the ratio of the probability of success and the probability of failure_.  

Odds are expressed in the form $X$ to $Y$, where $X$ and $Y$ are numbers. Usually, the word "to" is replaced by a symbol for ease of use, conventionally either a slash(/) or hyphen(-), although a colon(:) is sometimes seen.  

#### Example of Odds

For a event, with $sucess$ probability of $p = 0.8$ and $failure$ probability of $q = 1 -p = 0.2$  

+ Odds of Success
  + $odds(success) = p/(1-p) = p/q = .8/.2 = 4$
  + or we can say _the odds of success are 4 to 1_
+ Odds of Failure is:  
  + $odds(failure) = q/p = .25$
  + or we can say _the odds of failure are 1 to 4_
+ reciprocals
  + odds of success and the odds of failure are just reciprocals of one another
  + 1/4 = .25 and 1/.25 = 4

### Odds Ratio

[Odds ratio, wikipedia](https://en.wikipedia.org/wiki/Odds_ratio)  

The odds ratio (OR) is a statistic defined as **the ratio of the odds of A in the presence of B** and **the odds of A without the presence of B**.

**If the OR is greater than 1, then A is considered to be associated with B** in the sense that, compared to the absence of B, the presence of B raises the odds of A. Note that _this does not establish that B causes A_. Often the odds ratio is used to compare the occurrence of some outcome (A) in the presence of some exposure (B), with the occurrence of the outcome (A) in the absence of a particular exposure (absence of B).  

#### Example of Odds Ratio

Suppose that 7 out of 10 males are admitted to an engineering school while 3 of 10 females are admitted.

| Probs of:  | Admitted | Not-Admitted |
|:----------:|:--------:|:------------:|
| **Male**   | 0.7      | 0.3          |
| **Female** | 0.3      | 0.7          |

+ odds of admission for males: $odds(male) = .7/.3 = 2.33333$  
+ odds of admission for females: $odds(female) = .3/.7 = .42857$  
+ **odds ratio** for admission: $OR = 2.3333/.42857 = 5.44$

We can say, _for a male, the odds of being admitted are 5.44 times as large as the odds for a female being admitted_.

### Maximum likelihood estimation

[Maximum likelihood estimation, wikipedia](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation)  

In statistics, maximum likelihood estimation (MLE) is **a method of estimating the parameters of a statistical model, given observations**. MLE attempts to **find the parameter values that maximize the likelihood function**, given the observations. The resulting estimate is called a maximum likelihood estimate, which is also abbreviated as MLE.  

As an example, suppose that we are interested in the heights of adult female penguins, but are unable to measure the height of every penguin in a population (due to cost or time constraints). _Assuming that the heights are normally distributed with some unknown mean and variance, the mean and variance can be estimated with MLE while only knowing the heights of some sample of the overall population_. MLE would accomplish that by taking the mean and variance as parameters and finding particular parametric values that make the observed results the most probable given the normal model.  

MLE is a method in statistics for **estimating parameter(s) of a model for given data**. The basic intuition behind MLE is that **the estimate which explains the data best, will be the best estimator**.  

The main advantage of MLE is that it has **asymptotic property**. It means that **when the size of the data increases, the estimate converges faster towards the population parameter**. We use MLE for many techniques in statistics to estimate parameters. I have explained the general steps we follow to find an estimate for the parameter.

MLE Steps:  

1. **Assumption**
    + Make an assumption about the data generating function
2. **Likelihood Function**
    + formulate the likelihood function for the data, using the data generating function
    + function is the probability of observing this data given the parameters, i.e. $P(D|\theta)$
      + parameters $\theta$ depend on our assumptions and the data generating function.
3. **Estimation**
    + find an estimator for the parameter using optimization technique
    + by finding the estimate which maximizes the likelihood function

#### maximum likelihood estimate

$\hat{\theta} \in \{{ \underset{\theta \in \Theta} {\operatorname {arg max} }}\ { \mathcal {L}}(\theta \,;x)\}$, if a maximum value exists

+ ${ \mathcal {L}}(\theta \,;x)$, the likelihood function
  + given a statistical model, i.e. a family of distributions $\{f(\cdot \,;\theta )\mid \theta \in \Theta\}$
  + where $\theta$ denotes the (possibly multi-dimensional) parameter for the model

The method of maximum likelihood **finds the values of the model parameter, $\theta$ , that maximize the likelihood function, ${\mathcal {L}}(\theta \,;x)$**

#### Log-Likelihood

To work with the _natural logarithm of the likelihood function_, called the log-likelihood:

$\ell (\theta; x)=\ln {\mathcal {L}}(\theta; x)$

+ $\ell$ for log-likelihood
+ $\ln$ for natural logarithm

#### Average Log-Likelihood

Or o work with the _average_ log-likelihood:  

${\hat {\ell }}(\theta \,;x)={\frac {1}{n}}\ln {\mathcal {L(\theta \,;x)}}$

+ $\hat {\ell }$ for average log-likelihood
+ $n$ for number of observations

#### Example 1 of MLE

We have tossed a coin $n$ times and observed $k$ heads.  
Note that we consider head is success and tail is failure.  

1. (Assumption) The coin follows Bernoulli distribution function.
2. (Likelihood Function):
    + likelihood function: binomial distribution function $P(D|\theta)$
    + to find the best estimate for $p$ (Probability of getting head) given that $k$ of $n$ tosses are Heads
3. (Estimation): M- estimator is: $\hat{P} = \frac{k}{n}$

## Appendix: References

General:  

+ [What is the "intuitive" logic behind logistic regression?](https://www.quora.com/What-is-the-intuitive-logic-behind-logistic-regression)
+ [Why is logistic regression considered a linear model?](https://www.quora.com/Why-is-logistic-regression-considered-a-linear-model)
+ [Regression, Logistic Regression and Maximum Entropy](http://ataspinar.com/2016/03/28/regression-logistic-regression-and-maximum-entropy/)
+ [Regression, Logistic Regression and Maximum Entropy part 2 (code + examples)](http://ataspinar.com/2016/05/07/regression-logistic-regression-and-maximum-entropy-part-2-code-examples/)

Link Function:  

+ [Purpose of the link function in generalized linear model](https://stats.stackexchange.com/questions/48594/purpose-of-the-link-function-in-generalized-linear-model)
+ [What is the difference between a “link function” and a “canonical link function” for GLM](https://stats.stackexchange.com/questions/40876/what-is-the-difference-between-a-link-function-and-a-canonical-link-function/40880)
+ [The Difference Between Link Functions and Data Transformations](https://www.theanalysisfactor.com/the-difference-between-link-functions-and-data-transformations/)
+ [Problem understanding the logistic regression link function](https://stats.stackexchange.com/questions/80611/problem-understanding-the-logistic-regression-link-function)

Logit Function:  

+ [What is a Logit Function and Why Use Logistic Regression?](https://www.theanalysisfactor.com/what-is-logit-function/)
+ [What is the difference between logistic and logit regression?](https://stats.stackexchange.com/questions/120329/what-is-the-difference-between-logistic-and-logit-regression)

Odds and Odds Ratio:  

+ [How do I interpret odds ratios in logistic regression? | Stata FAQ](https://stats.idre.ucla.edu/stata/faq/how-do-i-interpret-odds-ratios-in-logistic-regression/)

MLE:  

+ [How do you explain maximum likelihood estimation intuitively?](https://www.quora.com/How-do-you-explain-maximum-likelihood-estimation-intuitively)
