# About Support Vector Machine

<!-- markdownlint-disable MD033 -->

<style>
p { text-indent: 5%; }
li { margin-left: -15px; }
color1 { color: crimson; }
</style>

## Intuitions of SVM

[Support-vector machine, wikipedia](https://en.wikipedia.org/wiki/Support-vector_machine)  

Given a set of training examples, each marked as belonging to one or the other of two categories, an SVM training algorithm builds a model that _assigns new examples to one category or the other_, making it a **_non-probabilistic binary linear classifier_**.  

An SVM model is <color1>a representation of the examples as points in space, mapped so that the examples of the separate categories are divided by a clear gap that is as wide as possible</color1>. New examples are then mapped into that same space and predicted to belong to a category based on which side of the gap they fall.  

In addition to performing linear classification, SVMs can efficiently perform a non-linear classification using what is called the kernel trick, implicitly mapping their inputs into high-dimensional feature spaces.  

<color1>**To Maximize margin**: we want to find the classifier whose decision boundary is furthest away from any data point.We can express the separating hyper-plane in terms of the data points that are closest to the boundary. And these points are called support vectors.</color1> We would like to learn the weights that maximize the margin. So we have the hyperplane!

### Advantages

+ **Effective in high dimensional spaces**
+ Still effective in cases where _number of dimensions is greater than the number of samples_
+ Uses a subset of training points in the decision function (called support vectors), so it is also **memory efficient**
+ Versatile: different **kernel functions can be specified for the decision function**. Common kernels are provided, but it is also possible to specify custom kernels.

### Disadvantages

+ If the number of features is much greater than the number of samples, avoid over-fitting in choosing Kernel functions and regularization term is crucial
+ SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation

## Steps of SVM Algorithm

### Linear Separable data

In the linear separable case, infinite decision boundaries are possible. The boundary that gives the maximum margin to the nearest observation is called the **optimal hyperplane**. The optimal hyperplane ensures the fit and robustness of the model. To find the optimal hyperplane, use the equation: $w \cdot X + b = 0$  

1. Prepare Dataset and Variables initializations
    + sample data are in 2D and with two labels
    + for convenience, class A is labelled as $+1$, class B is labelled as $-1$
2. Initialize variables for optimization searching
    + $w$, ranging from large to small
      + remember to transform $w$, i.e. $[1, 1], [1, -1], [-1, 1], [-1, -1]$
      + in 2D, $x_i$ can be in any of the four quadrants, this may cause problem if not transformed
      + or, if data is always in 1st quadrant, transformation is not needed
    + $b$, ranging from large to small
3. Optimization to find the optimal parameters
    + iterate through all the data $(x_i, y_i)$
    + if $y_i \times (w \cdot x_i - b) \ge 1$, the parameters correctly separate all the data points
      + $> 1$, data is one of the two classes, not support vectors
      + $= 1$, data is a support vector, on a line (in 2D)
    + if $y_i \times (w \cdot x_i - b) < 1$, the parameters cannot separate all the data points
4. After all iterations
    + use the parameters $w$ and $b$ that have minimial $||w||$
    + `numpy.linalg.norm` can return the magnitude of a vector
5. Optimization goal is to
    + **find $w$ and $b$ so that the margin is maximized, $\frac{2}{||w||}$, i.e. minimize $||w||$**
    + or, identify the support vectors that satisfying $y_i \times (w \cdot x_i - b) = 1$

### Non Separable data

## Backgrounds of SVM

1. Equations for line, plane and hyperplane
2. Vectors
    + Dot Product
    + Orthogonal Projection
    + Unit Vector
3. Margin of two hyperplanes
4. Hard Margin vs. Soft Margin and Hinge-loss, Slack Variable
5. Primal and Dual Problems
6. Kernel Tricks

<!-- 6. Quadratic Programming
7. Lagrangian constraint optimization -->

### Equations: Line, Plane, Hyperplane

[Plane Equation, wikipedia](http://www.songho.ca/math/plane/plane.html)  

+ Line Equation, in 2-D
  + $y = m \cdot x + d$, is the same as $ax + by+ c = 0$
    + pf: $y = (\frac{-a}{b}) \cdot x + (\frac{-c}{b})$, i.e. $m = (\frac{-a}{b}), d = (\frac{-c}{b})$
  + can be written as: $ax_1 + b_2x_2 + c = 0$, there may be more than 26 dimensions
  + or as: $w_1x_1 + w_2x_2 + w_0 = 0$
+ Plane Equation, in 3-D
  + can be written as: $ax + by + cz + d = 0$
  + or as: $w_1x_1 + w_2x_2 + w_3x_3 + w_0 = 0$
+ Hyperplance Equation, in n-D
  + $w_1x_1 + w_2x_2 + w_3x_3 + ... + w_nx_n + w_0 = 0$
    + $n$ for the number of dimensions
  + can be shortened to: $w_0 + \sum_{i=1}^{n}{W_iX_i} = 0$
    + where $W_i$ is $[w_1, w_2, ..., w_n]$
    + where $X_i$ is $[x_1, x_2, ..., x_n]$
  + with _vector notation_: $w_0 + W_{1 \times n} \cdot X_{n \times 1} = 0$
    + $W$ for a vector of (1-row, n-col), having coefficients $[w_1, w_2, ..., w_n]$
    + $X$ for a vector of (n-row, 1-col), having $[x_1, x_2, ..., x_n]$
  + with _vector notation_ and _transpose_: $w_0 + W^T \cdot X = 0$
    + $W$ for a vector of (n-row, 1-col), having coefficients $[w_1, w_2, ..., w_n]$
    + $ W^T$ for transpose of $W$
    + $X$ for a vector of (n-row, 1-col), having $[x_1, x_2, ..., x_n]$

### Vectors

[Euclidean_vector, wikipedia](https://en.wikipedia.org/wiki/Euclidean_vector)  
In mathematics, a Euclidean vector (or just "vector") is a geometric object that has **magnitude (or length) and direction**. A vector is what is needed to "carry" the point A to the point B; the Latin word vector means "carrier".  
A Euclidean vector is frequently represented by a line segment with a definite direction, or graphically as an arrow, **connecting an initial point O with a terminal point A, and denoted by ${\vec{OA}}$**. **The magnitude or length of a vector $\vec{u}$ (O-A distance) is written as $|u|$ (sometimes it's written as $∥u∥$, to avoid confusion between scalar and vector)** and called its norm. The direction of the vector $\vec u$ is defined by the angle $\theta$ with respect to the horizontal axis, and by the angle $\alpha$ with respect to the vertical axis.  

+ On a 2-Dimension plane, with horizontal axis: dimension_1 and vertical axis: dimension_2
  + if $o:(0, 0)$ and $a:(3, 4)$
    + $\vec{oa}$ is moving from $o$ to $a$, or we can shortened as $\vec A = [3, 4]$
  + (direction) with 2 dimensions, it's saying that $\vec A$ is:
    + moving 3 units in dimenion 1
    + and moving 4 units in dimension 2
  + (magnitude) with 2 dimensions, it's Euclidean distance from $o$ to $a$:
    + $||\vec A|| = 5$

#### Dot Product of Vectors

[Dot_product, wikipedia](https://en.wikipedia.org/wiki/Euclidean_vector#Dot_product)  
The dot product of two vectors $a$ and $b$ is denoted by $\vec a \cdot \vec b$, can be computed by the multiplying the magnititude of both vectors and $\cos{\theta}$, where $\theta$ is the angle between the two vectors, or computed by _the sum of the products of the components of each vector_:  

1. $\vec a \cdot \vec b = ||\vec a|| \; ||\vec b|| \cos{\theta}$
    + where $\theta$ is the measure of the angle between $a$ and $b$
    + recall that $\cos{0} = 1, \cos{60} = 1/2, \cos{90} = 0$
2. $\vec a \cdot \vec b =a_{1}b_{1}+a_{2}b_{2}+a_{3}b_{3}$
    + where $\vec a$ and $\vec b$ are both vectors in 3 dimensions
    + i.e. $\vec a = [a_1, a_2, a_3], \vec b = [b_1, b_2, b_3]$

+ On a 2-Dimension plane, with horizontal axis: dimension_1 and vertical axis: dimension_2
+ if $\vec A = [1, 3]; \vec B = [4, 2]$
  + dot product of $\vec A$ and $\vec B$: $\vec A \cdot \vec B = (1 \times 4) + (3 \times 2) = 10 $

#### Unit Vector

[Unit vector, wikipedia](https://en.wikipedia.org/wiki/Unit_vector)  
&nbsp;&nbsp;&nbsp;&nbsp;The normalized vector or versor û of a non-zero vector $u$ is the unit vector in the direction of $u$, i.e., $\mathbf {\hat {u}} ={\cfrac {\mathbf {u} }{|\mathbf {u} |}}$, where $|u|$ is the norm (or length) of $u$.  
**A unit vector is a vesctor which has a magnitude of 1**. The term **normalized vector** is sometimes used as a synonym for unit vector.  

#### Orthogonal Projection of a Vector

[Vector_projection, wikipedia](https://en.wikipedia.org/wiki/Vector_projection)  
<p align='center'>
<img src='./references/figures/orthogonal_projections.jpeg'>
<em>Vector x projected onto vector y (Left), new vector z(right); we can get the distanced between two vectors: $||x||$ and $||y||$ by projecting $$||x||$ to $||y||$, the distanced is x-z. </em>
</p>

### Margin of two hyperplanes

<p align='center'> <img src='./references/figures/maximize_margin.png'> </p>

Prove the margin width is $\frac{2}{||\vec w||}$:  

+ Let $x_1$ be on the hyperplane $\vec wx - b = -1$, $x_2$ be on $\vec wx - b = 1$
+ margin between the two hyperplanes be $r$
+ $r$ is the perpendicular distance from $x_1$ to $\vec wx - b = 1$
+ $\frac{\vec w}{||\vec w||}$ be the unit vector of $\vec w$ with $magnitude=1$, or $length=1$
  + $w(x_1 + r \times \frac{w}{||w||}) - b = 1$
    + move $x_1$ by the magnitude of the margin $r$ will land on another hyperplane
  + $\iff wx_1 + r \times \frac{w \times w}{||w||} - b = 1 \iff wx_1 + r \times \frac{||w||^2}{||w||} - b = 1$
    + recall that vector has direction and magnitude, squared doesn't change direction
  + $\iff wx_1 + r \times ||w|| - b = 1 \iff wx_1 - b = 1 - r \times ||w||$
    + since $x_1$ on $wx - b = -1$, $wx_1 - b = -1$
  + $\iff -1 = 1 - r \times ||w|| \iff r = \frac{2}{||w||}$

### Hard-Margin vs. Soft-Margin and Hinge-Loss

#### Hard-margin

[Hard-margin, wikipedia](https://en.wikipedia.org/wiki/Support-vector_machine#Hard-margin)  
If the training data is _linearly separable_, we can select two parallel hyperplanes that separate the two classes of data, so that the distance between them is as large as possible. The equations for these two hyperplanes are:  

1. $\vec w \cdot \vec x - b = 1$, anything above is labelled as class +1
2. $\vec w \cdot \vec x - b = -1$, anything below is labelled as class -1

The distance between these two hyperplanes is $\frac{2}{||\vec w||}$, and the optimization goal is to maximize the margin (width), i.e. minimize $||\vec w||$. **Another constraint $y_i \times (\vec w \cdot \vec x_i - b) \ge 1$ is set to _prevent_ any datapoint to fall within the margin**. Hence, the optimization goal is to:  

+ Minimize $||\vec w||$ subject to $y_i (\vec w \cdot \vec x_i - b) \ge 1$, for $i = 1... n$

Note that **with hard-margin, the hyperplane is entirely determined by $\vec x$ that is closest to it**, i.e. the support vectors. It's likely causing over-fitting if data is not linearly separable.  

#### Soft-margin

[Soft-margin, wikipedia](https://en.wikipedia.org/wiki/Support-vector_machine#Soft-margin)  
Soft-margin SVM is an extension on hard-margin, with **hinge-loss function** introduced. Intuitively, hinge-loss is to penalize mis-classfications, i.e. data is on the wrong side of the hyperplane. If $\vec x_i$ lies on the correct side of the margin, hinge-loss is zero.  
**For data on the wrong side of the margin, the hinge-loss value is proportional to the distance from the margin, i.e. for wrong data, the farther from the margin, the larger the hinge loss**. There're two ways to introduce the optimization goal:  

1. Minimize $\lambda\lVert \vec w \rVert^2 + \frac 1 n \sum_{i=1}^n \max\left(0, 1 - y_i(\vec w x_i + w_0)\right)$
    + $\lambda$ for controlling the tradeoff between margin size and $\vec x_i$ on the right side
      + if $\lambda$ is sufficiently small, $\lambda {||\vec w||}^2$ is negligible
        + then the optimization goal becomes minimize the hinge-loss term
        + i.e. the less mis-classifications, the better
        + i.e. will behave similar to hard-margin SVM (if linearly separable)
2. or Minimize $\frac{1}{2} \lVert \vec w \rVert^2 + C \sum_{i=1}^n \xi_i$, where  $\xi_i \geq 0, y_i (\vec w x_i + w_0) \geq 1-\xi_i, \forall i$
    + $\xi$ is called the _slack variable_
      + to allow some datapoints to be on the wrong side of the margin
      + if $\xi_i = 0$, meaning $x_i$ is correctly classified
      + if $\xi_i > 0$, meaning $x_i$ is on the wrong side of the margin
      + if $\xi_i > 1$, meaning $x_i$ is on the wrong side of the hyperplane
        + this is 'more' wrong than $> 0$, therefore penalized more
    + $C$ is called _budget_, a tuning parameter
      + to control how many datapoints can be on the wrong side
      + to controls how much the individual $\xi_i$ can be modified to violate the margin
      + if $C = 0$, it's similar to hard-margin SVM
      + as $C$ increases, margin widens and more mis-classifications
3. The relationship between $C$ and $\lambda$ is: $C = \frac{1}{\lambda}$, or more precisely: $C = \frac{1}{2n\lambda}$

The second forumulation removes the non-differentiable $\max$ function and allows the optimization to be solved using quadratic programming, it's called _primal problem_ of SVM.  

#### Hinge Loss

[Hinge-loss, wikipedia](https://en.wikipedia.org/wiki/Hinge_loss)  
The hinge loss is used for "maximum-margin" classification, most notably for support vector machines. For an intended output $y_i = [+1, -1]$ and a classifier score $y, y \in R$, the hinge-loss of the prediction $y$ is defined as:  

+ $\ell(y) = \max(0, 1 - y_i \cdot y)$
  + $y_i = [+1, -1]$, i.e. the label for each $i$ observation
  + $y = \vec w \cdot \vec x_i - b$, i.e. the raw output of decision function, not $[+1, -1]$ label
  + recall from SVC, if $y_i (\vec w \cdot \vec x_i - b) \ge 1$, it's **correctly classified**
    + i.e. if $y_i$ and $y$ have the same sign and $y_i \cdot y \ge 1$, $\ell(y) = \max(0, < 0) = 0$
    + i.e. **hinge-loss = 0**
  + when $y_i = +1$ and $1 > y > 0$
    + (same sign) e.g. $y = 0.9$, data is within the margin but near $+1$ hypeplane
      + $\ell(y) = \max(0, 1- 1 \times 0.9) = 0.1$
    + (diff. sign) e.g. $y = -0.9$, data is within the margin but near $-1$ hypeplane
      + $\ell(y) = \max(0, 1 - (1 \times -0.9)) = \max(0, 1.9) = 1.9$
    + i.e. **the farther from correct hyperplane, the larger the hinge-loss**
  + when $y_i = -1$ and $-1 < y < 0$
    + similar to when $y_i = +1$ and $1 > y > 0$

Intuitively, hinge-loss is to penalize the mis-classifications. If data is correctly classified, the hinge-loss is just zero. **But if data is mis-classified (have different sign), the hinge-loss will increses with the the datapoint's distance from correct hyperplane.** In other words, when $y_i$ and $y$ have opposite signs, hinge loss increases linearly with $y$. Overall, we want to have _less mis-classifications_, i.e. _lower hinge-loss_.  

#### Slack Variable

[Slack variable, wikipedia](https://en.wikipedia.org/wiki/Slack_variable)  
In an optimization problem, a slack variable is a variable that is added to an inequality constraint to transform it into an equality. **Introducing a slack variable replaces an inequality constraint with an equality constraint and a non-negativity constraint on the slack variable**.  

##### Example of Slack Variable

By introducing the slack variable $y \ge 0$, the inequality $A x \le b$ can be converted to the equation $A x + y = b$.

### Primal and Dual Problems

#### Primal

[Primal, wikipedia](https://en.wikipedia.org/wiki/Support-vector_machine#Primal)  
The primal problem is to find the normal vector $\vec w$ and the bias $b$.

#### Dual

[Dual, wikipedia](https://en.wikipedia.org/wiki/Support-vector_machine#Dual)  
The dual problem is to express $\vec w$ as a linear combination of the training data $x_i$, i.e. $\vec w = \sum_{i=1}^m \alpha_i y_i \mathbf{x}_i$, where $y_i \in \{1, -1\}$ represents the class of the training example and $\alpha_i$ are Lagrange multipliers.  
$\max \sum_{i=1}^m \alpha_i - \frac{1}{2} \sum_{i,j=1}^{m}y_iy_j\alpha_i\alpha_j \langle x_i, x_j \rangle$  

### Kernel Tricks

[Kernel method, wikipedia](https://en.wikipedia.org/wiki/Kernel_method)  
[The Kernel Trick, quora](https://dscm.quora.com/The-Kernel-Trick)
In its simplest form, the kernel trick means transforming data into another dimension that has a clear dividing margin between classes of data. Kernel methods require only a user-specified kernel, i.e., a **similarity function** over pairs of data points in raw representation.  
Kernel methods owe their name to the use of kernel functions, which enable them to operate in a _high-dimensional, implicit feature space_ without ever computing the coordinates of the data in that space, but rather **by simply computing the inner products between the images of all pairs of data in the feature space**. This operation is often computationally cheaper than the explicit computation of the coordinates. This approach is called the _"kernel trick"_.  

#### Trick with Inner Product

<p align='center'>
<img src='./references/figures/kernel_tricks_1.jpg'>
</p>

+ Original data in 2D, not linearly separable, denoted as $X = \{x_1, x_2\}$
+ Transformed data in 3D, can be separated
  + transform function $\Phi$, a function from 2D to 3D
  + transformed by $\Phi(X) \rightarrow x_1^2, x_2^2, \sqrt{2}x_1x_2$
  + new decision function: $\beta_0 + \beta_1x_1^2 + \beta_2x_2^2 + \beta_3\sqrt{2}x_1x_2 = 0$

To transform data using kernel, we could do it manually, or using the property inner product.  

+ Do it completely, i.e. transform data: $\langle \Phi(x_i), \Phi(x_j) \rangle$
  + $= \langle \{ x_{i1}^2, x_{i2}^2, \sqrt{2}x_{i1}x_{i2} \}, \{ x_{j1}^2, x_{j2}^2, \sqrt{2}x_{j1}x_{j2} \}\rangle$
    + apply transform function, $3 \times 2$ operations
  + $= x_{i1}^2x_{j1}^2 + x_{i2}^2x_{j2}^2 + 2 \cdot x_{i1}x_{i2}x_{j1}x_{j2}$
    + inner products, $3$ operations
+ Use inner product property: $\langle x_i, x_j \rangle^2$
  + $= \langle \{ x_{i1}, x_{i2} \}, \{ x_{j1}, x_{j2} \} \rangle^2$
    + this has no operation in computation
  + $= (x_{i1}x_{j1} + x_{i2}x_{j2})^2$
    + inner product, with $2$ operations
  + $= x_{i1}^2x_{j1}^2 + x_{i2}^2x_{j2}^2 + 2 \cdot x_{i1}x_{i2}x_{j1}x_{j2}$
    + expand the square, $1$ operation

By just computing the dot product in the original space and raising the result (a scalar) to a power, we can get the same results but with less operations. In the previous example, the kernel function is (linear): $K(x_i, x_j) = (a \langle x_i, x_j \rangle + b)^n$.  
In short, the Kernel trick is a 'trick' because that **mapping does not need to be ever computed**. **If our algorithm can be expressed only in terms of a inner product between two vectors, all we need is replace this inner product with the inner product from some other suitable space**. That is where resides the “trick”: wherever a dot product is used, it is replaced with a Kernel function.  

#### Gaussian Kernel, or Radial Basis Function

[Radial basis function kernel, wikipedia](https://en.wikipedia.org/wiki/Radial_basis_function_kernel)  
[RBF SVM parameters, sklearn](https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html)
$$K_{\text{Gauss}}(\mathbf{x}_i, \mathbf{x}_j) = e^{\frac{\|\mathbf{x}_i - \mathbf{x}_j\|^2}{2 \sigma^2}} = \exp({\frac{\|\mathbf{x}_i - \mathbf{x}_j\|^2}{2 \sigma^2}}) = \exp({-\gamma\|\mathbf{x}_i - \mathbf{x}_j\|^2}) $$

Theoretically, the gaussian kernel has"infinite" number of dimensions.  

+ $\|x_i - x_j\|^2$ is the squared Euclidean distance between the two feature vectors
  + if $x_i$ and $x_j$ is close, $\|x_i - x_j\|^2$ value will be smaller, i.e. more closer
+ $\sigma$ is a free parameter
+ $\gamma = \frac{1}{2 \sigma^2}$, controlling the effect of distance between the pair
  + note $\exp({-\gamma\|\mathbf{x}_i - \mathbf{x}_j\|^2}) = \frac{1}{\exp({\gamma\|\mathbf{x}_i - \mathbf{x}_j\|^2})}$
  + with larger $\gamma$, closer pair has higher influence, farther has lower
  + with smaller $\gamma$, the farther distance between pair has increased influence
+ note that $\exp(0) = 1$ and $\exp(1) = 2.7183$
+ its value decreases with distance, i.e. larger distance, smaller RBF value
+ its range is between 0 (limit of $\exp$) and 1 (when $x_i = x_j$, is $\exp(0)$)


#### Inner Product

[Inner product space, wikipedia](https://en.wikipedia.org/wiki/Inner_product_space)  
An inner product is _a generalization of the dot product_. In a vector space, **it is a way to multiply vectors together, with the result of this multiplication being a scalar**. More precisely, for a real vector space, an inner product $\langle\cdot,\cdot\rangle$ has the following four properties.  

+ Let $\vec u$, $\vec v$, and $\vec w$ be vectors and $\alpha$ be a scalar, then:
  1. $\langle \vec u + \vec v, \vec w \rangle = \langle \vec u, \vec w \rangle + \langle \vec v, \vec w\rangle$
  2. $\langle \alpha \vec v, \vec w \rangle = \alpha \langle \vec v, \vec w \rangle$
  3. $\langle \vec v, \vec w \rangle = \langle \vec w, \vec v \rangle$
  4. $\langle \vec v, \vec v \rangle \ge 0$ and $\langle \vec v, \vec v \rangle = 0$ if and only if $\vec v = 0$

In linear algebra, an **inner product space** is a vector space with an additional structure called an inner product. This additional structure associates each pair of vectors in the space with a scalar quantity known as the inner product of the vectors.  

+ Inner Producgt of real numbers, $x, y \in R$
  + $\langle x, y \rangle := xy$
  + the inner product of $x$ and $y$, both are real numbers, is defined as $x \times y$
+ Inner Product in Euclidean space, $\vec x, \vec y \in R^n$
  + Euclidean space is an n-dimensional space that is defined in Euclidean geometry
  + Euclidean inner product is actually a dot product or scalar product of two n-D vectors
  + $\langle \vec x, \vec y \rangle := \sum_{i=1}^{n}\ x_{i}y_{i}=x_{1}y_{1}+x_{2}y_{2}+...+x_{n}y_{n} $

##### Example of Inner Product

Example1:  
Evaluate the inner product of vectors $\vec a = (3, 4, -1)$ and $\vec b = (7, -2, 3)$  
$\langle \vec a, \vec b \rangle = a_1b_1 + a_2b_2 + a_3b_3 = 21 - 8 -3 = 10$ units  

### Quadratic Programming

[Quadratic programming, wikipedia](https://en.wikipedia.org/wiki/Quadratic_programming)  
Quadratic programming (QP) is the process of solving a special type of mathematical optimization problem—specifically, a (linearly constrained) quadratic optimization problem, that is, the problem of optimizing (minimizing or maximizing) a quadratic function of several variables subject to linear constraints on these variables. Quadratic programming is a particular type of nonlinear programming.  

## Appendix: References

General:  

+ [12: Support Vector Machines (SVMs)](http://www.holehouse.org/mlclass/12_Support_Vector_Machines.html)
+ [Using SVMs with sklearn](https://martin-thoma.com/svm-with-sklearn/)
+ [How does a Support Vector Machine (SVM) work?](https://stats.stackexchange.com/questions/23391/how-does-a-support-vector-machine-svm-work)
+ [Support Vector Machine introduction, pythonprogramming](https://pythonprogramming.net/support-vector-machine-intro-machine-learning-tutorial/)
+ [Beginning SVM from Scratch in Python, pythonprogramming](https://pythonprogramming.net/svm-in-python-machine-learning-tutorial/)
+ [Support Vector Machine Parameters, pythonprogramming](https://pythonprogramming.net/support-vector-machine-parameters-machine-learning-tutorial/?completed=/soft-margin-kernel-cvxopt-svm-machine-learning-tutorial/)
+ [Support Vector Machines: A Guide for Beginners](https://www.quantstart.com/articles/Support-Vector-Machines-A-Guide-for-Beginners)
+ [Support Vector Machines, wikibooks](https://en.wikibooks.org/wiki/Support_Vector_Machines)
+ [Support Vector Machines, sklearn](https://scikit-learn.org/stable/modules/svm.html)
+ [Chapter 3: Support Vector machine with Math, medium](https://medium.com/deep-math-machine-learning-ai/chapter-3-support-vector-machine-with-math-47d6193c82be)
+ [Support vector machines ( intuitive understanding ) — Part#1](https://towardsdatascience.com/support-vector-machines-intuitive-understanding-part-1-3fb049df4ba1)

Vectors:  

+ [What is the use of the double modulus signs?](https://math.stackexchange.com/questions/1302937/what-is-the-use-of-the-double-modulus-signs)
+ [What happens when a vector is divided by its magnitude?](https://www.quora.com/What-happens-when-a-vector-is-divided-by-its-magnitude)

Equations:

+ [Linear Algebra | Equation of a line (2-D) | Plane(3-D) | Hyperplane (n-D), youtube](https://www.youtube.com/watch?v=3qzWeokRYTA)  

Margin:

+ [Why does the SVM margin is $\frac{2}{∥w∥}$](https://math.stackexchange.com/questions/1305925/why-does-the-svm-margin-is-frac2-mathbfw)

Hinge-loss, Hard-Margin, Soft-Margin:  

+ [What is the difference between C and lambda in the context of an SVM?](https://datascience.stackexchange.com/questions/22408/what-is-the-difference-between-c-and-lambda-in-the-context-of-an-svm)
+ [What are the objective functions of hard-margin and soft margin SVM?](https://www.quora.com/What-are-the-objective-functions-of-hard-margin-and-soft-margin-SVM)
+ [What is the loss function of hard margin SVM?](https://stats.stackexchange.com/questions/74499/what-is-the-loss-function-of-hard-margin-svm)

Primal and Dual Problems:  

+ [Why bother with the dual problem when fitting SVM?](https://stats.stackexchange.com/questions/19181/why-bother-with-the-dual-problem-when-fitting-svm)

Kernel Tricks and Inner Product:  

+ [Inner Product](http://mathworld.wolfram.com/InnerProduct.html)
+ [Inner Product](https://math.tutorvista.com/algebra/inner-product.html)
+ [What is the kernel trick?](https://www.quora.com/What-is-the-kernel-trick)
+ [Kernels Introduction](https://pythonprogramming.net/kernels-with-svm-machine-learning-tutorial/)
+ [How to intuitively explain what a kernel is?](https://stats.stackexchange.com/questions/152897/how-to-intuitively-explain-what-a-kernel-is/152904)
+ [Kernels and Feature maps: Theory and intuition](https://xavierbourretsicotte.github.io/Kernel_feature_map.html)
