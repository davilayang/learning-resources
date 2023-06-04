# Linear Regression

## Intuitions of Linear Regression

[Linear regression, wikipedia](https://en.wikipedia.org/wiki/Linear_regression#Machine_learning)  
&nbsp;&nbsp;&nbsp;&nbsp; The very simplest case of a single scalar predictor variable $x$ and a single scalar response variable $y$ is known as **_simple linear regression_.** The extension to multiple and/or vector-valued predictor variables (denoted with a capital $X$) is known as **_multiple linear regression_**, also known as _multivariable linear regression._
Note, however, that in these cases the response variable y is still a scalar. Another term, _multivariate linear regression_, refers to cases where $y$ is a vector, i.e., the same as general linear regression.  
&nbsp;&nbsp;&nbsp;&nbsp; The liner regression model takes form:  

+ With CS909 Data Mining:
  + $y = w_0 + w_1 \cdot x_1 + w_2 \cdot x_2 + \cdots + w_n \cdot x_n + \epsilon_i$
    + $w_0$ for the intercept
    + $w_1 \cdot x_1 + \cdots + w_n \cdot x_n$ for Partial Regression Coefficients
    + $\epsilon_i$ for the residuals
  + Using vector: $y = \sum_{j=1}^{d} x_{ij}w_j = \boldsymbol{Xw}$, where $x_{i1}=1$
    + $d$ for number of input dimensions
+ With CS229 by Andrew Ng:
  + $h_{\theta}(x) = h(x) = \theta_0 + \theta_1 x_1 + \cdots + \theta_n x_n$
  + Using vector: $h(x) = \sum_{i=0}^{n} \theta_i x_i = \theta^T x$, where $x_0 = 1$
    + $n$ for number of input variables, excluding $x_0$

## Steps of Linear Regression Algorithm

## Backgrounds of Linear Regression

1. $R^2$ and Adjusted $R^2$
2. Ordinary Least Square (OLS) and Cost/Loss function

### $R^2$ and Adjusted $R^2$

+ Regular $R^2$
  + Coefficient of Determnination
  + _**NOT** account for the number of predictors used_
  + $= \cfrac{\text{variation accounted for by $x$ variables}}{\text{total variation}}$
  $= 1 - \cfrac{\text{variation NOT accounted for by $x$ variables}}{\text{total variation}}$
+ Adjusted $R^2$, or $R^2_{adj}$
  + _Add a **penalty on the number of predictors used**_
  + $= 1 - \cfrac{n-1}{n-p-1}(1 - R^2)$
    + $p$ for the number of predictors used (exclude the constant term)
    + $n$ for the sample size, or number of observations
  
### Ordinary Least Square (OLS)

[Ordinary least squares, wikipedia](https://en.wikipedia.org/wiki/Ordinary_least_squares)  
&nbsp;&nbsp;&nbsp;&nbsp; OLS chooses the parameters of a linear function of a set of explanatory variables by the principle of least squares: _minimizing the sum of the squares of the differences between the observed dependent variable (values of the variable being predicted) in the given dataset and those predicted by the linear function_.  

+ Cost/Loss Function with CS229
  + $J(\theta) = \frac{1}{2} \sum_{i=1}^{n} (h_{\theta}(x^{(i)}) - y^{(i)} )^2$
  + $\frac{1}{2}$ is for easier partial differentiation later
+ Cost/Loss Function with CS909
  + $L(w) = \sum_{i=1}^{N} (y_i - \hat{y_i})^2$ $= (\boldsymbol{y - Xw})^T (\boldsymbol{y - Xw})$ $= \sum_{i=1}^{N} (y_i - x_i^T \boldsymbol{w})^2$

#### Partial Derivative on Cost/Loss Function

+ $\frac{\partial}{\partial \theta_j}J(\theta)$ $=\frac{\partial}{\partial \theta_j} \frac{1}{2} (h_{\theta}(x) - y)^2$
  + $= 2 \cdot \frac{1}{2} (h_{\theta}(x) - y) \cdot \frac{\partial}{\partial \theta_j} (h_{\theta}(x) - y)$
  + $= (h_{\theta}(x) - y) \cdot \frac{\partial}{\partial \theta_j} (h_{\theta}(x) - y)$
  + $= (h_{\theta}(x) - y) \cdot \frac{\partial}{\partial \theta_j} (\sum_{i=0}^{n} \theta_ix_i - y)$
  + $= (h_{\theta}(x) - y) \cdot x_i$
+ $\frac{\partial}{\partial w} L(w)$ $= \frac{\partial}{\partial w} (\boldsymbol{y - Xw})^T (\boldsymbol{y - Xw})$
  + $= \frac{\partial}{\partial w} \; y^Ty - y^TXw - X^Tw^Ty + X^Tw^TXw$
  + $= \frac{\partial}{\partial w} \; y^Ty - (y^TXw + X^Tw^Ty) + X^Tw^TXw$
  + $= \frac{\partial}{\partial w} \; y^Ty - (2y^TXw) + X^Tw^TXw$, (matrix transpose property)
  + $= 0 + 2X^TXw - 2X^Ty$
  + $= 2(X^TXw - X^Ty) = 2 (Xw - y) X^T$

#### LMS Update on Weights

Least Mean Square Update rule

+ $w_1 \leftarrow w_0 - \eta \frac{dL}{dw}|_{w = w_0} $
  + $\eta$ for learning rate
  + 
+ $\theta_j := \theta_j + \alpha \sum_{i=1}^{m} \big(y^{(i)} - h_{\theta}(x^{(i)}) \big) \cdot x_j^{(i)}$
  + $\alpha$ for learning rate
  + $m$ for the training examples/observations
  + $j$ for each attribute/dimension
  + on each step, update $\theta_j$ with every example ($m$) in the training set
    + "Batch Gradient Descent"

## Appendix: Reference

$\frac{\partial L(w)}{\partial w} = 0$  
$L(w)  = 0 - 22 + 28w = 0$  
$w = \frac{22}{28} = 0.7857$  
