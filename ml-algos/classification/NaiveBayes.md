# About Naive Bayes Classifier

## Intuitions of Naive Bayes Classifier

[Naive Bayes classifier](https://en.wikipedia.org/wiki/Naive_Bayes_classifier)  
&nbsp;&nbsp;&nbsp;&nbsp; In machine learning, naive Bayes classifiers are a family of simple "probabilistic classifiers" based on applying Bayes' theorem **with strong (naive) independence assumptions between the features**.  
&nbsp;&nbsp;&nbsp;&nbsp; There is not a single algorithm for training such classifiers, but a family of algorithms based on a common principle: **all naive Bayes classifiers assume that the value of a particular feature is independent of the value of any other feature, given the class variable**.  

### Model of Conditional Probability

+ Probability of a class conditional on given features
+ $= P(C_i | X)$, where $X = (x_1, x_2, x_3, ..., x_n)$
  + $C_i$ for each class/label in target attribute
  + $i$ for each class/label
    + e.g. if binary with $\{positive, negative\}$: $C_1$ is $postive$, $C_2$ is $negative$

The goal is to select the class/label $C_i$ with the highest probability, i.e. maximize $P(C_i \mid X)$. Or, the decision rule for Naive Bayes classification is MAP, Maximum a posterior.  

### Reformulated model

&nbsp;&nbsp;&nbsp;&nbsp; Problem with conditional probability model is that if number of features $n$ is large, it's impossible to compute $p(C_i | X)$. Therefore, the model is reformulated using _Baye's Theorem_:  

$$ P(C_i \mid X) = \cfrac{P(X \mid C_i) \times P(C_i)}{P(X)} $$

+ where the goal is to maximize $P(C_i \mid X)$
  + $P(X)$, proabability of features
    + it's not related to class/label $C_i$, treated as constant in practice
    + i.e. goal is just to maximize $P(X \mid C_i) \times P(C_i)$
  + $P(C_i)$ for the probability of class $k$, "_class prior probability_"
    + if not known, _the goal is to maximize_ $P(X \mid C_i) \times P(C_i)$
      + usually assumed to have equal probability
      + e.g. if binary classification, $P(C_i)$ is assumed to be $\frac{1}{2}$
    + if known, _the goal is just to maximize_ $P(X \mid C_i)$
    + can be estimated by $\frac{|C_{i, D}|}{|D|}$
      + (number of observations in class $k$) / (total number of observations)

In plain English: $(\text{posterior probability}) = \cfrac{(\text{likelihood}) \times (\text{prior probability)}}{\text{evidence}}$

+ $\text{posterior probability}$, i.e. $P(C_i \mid X)$
  + _posterior: the probabiltiy given the conditions_
  + our goal is to Maximum A Posteriori, i.e. MAP estimation
+ $\text{prior probability}$, i.e. $P(C_i)$
  + _prior: the probability before given any condition_
  + $P(C_i) = \frac{\text{number of observations belonging to $C_i$}}{\text{number of observations}}$
  + e.g. if binary classification, the probabiltiy of $positive$ and $negative$ class in dataset
+ $\text{likelihood}$, i.e. $P(X \mid C_i)$
  + given class/label, the probability of the features
  + assume conditionally independent $P(X|C_i) = P(x_1|C_i) \times P(x_2|C_i) \times... \times P(x_n|C_i)$
+ $\text{evidence}$ for the given "evidence", i.e. the probability of features $P(X)$
  + the probability of given evidence $X$ is fixed

### Naive Assumption

**The assumption: class-conditional independence**  
&nbsp;&nbsp;&nbsp;&nbsp; This presumes that the **attributes’ values are conditionally independent of one another, given the class label of the tuple** (i.e., that there are no dependence relationships among the attributes).  
&nbsp;&nbsp;&nbsp;&nbsp; Each feature $x_i$ is conditionally independent of every other feature $x_j$ ($i \ne j$), given the class/label $C_k$, e.g. $P(x_1, x_2 \mid C_i) = P(x_1 \mid C_i) \times p(x_2 \mid C_i)$. We can generalize the formula as:  

$$P(X | C_i) = \prod\limits_{k=1}^{n} (x_k \mid C_i) = P(x_1 | C_i) \times P(x_2 | C_i) \times ... \times P(x_n | C_i)$$  

For each attribute $A_k$, we look at whether the attribute is categorical or continuous-valued, ($D$ for the dataset).  

+ if $A_k$ is categorical
  + $P(x_k | C_i) = \cfrac{\text{# of observations of class $C_i$ in $D$ having value $x_k$ for $A_k$}}{\text{# of observations of class $C_i$ in $D$}}$
+ if $A_k$ is continuous-valued
  + usually assumed to have a _Gaussian Distribution_
    + with mean $\mu$ and standard deviations $\sigma$
    + its form is defined by: $g(x, \mu, \sigma) = \frac{1}{\sqrt{2 \pi} \cdot \sigma} \times \exp(-\frac{(x - \mu)^2}{2 \sigma^2})$
  + $P(x_k | C_i) = g(x_k, \mu_{C_i}, \sigma_{C_i})$
    + $\mu_{C_i}$ for the mean of the observations of $A_k$ of class $C_i$
    + $\sigma_{C_i}$ for the standard deviation of the observations of $A_k$ of class $C_i$
  + or using discretization
    + but discretization may throw away discriminative information

<!-- $P(X \mid C_i) \times P(C_i)$ can be reformulated as $P(C_i \cap X)$ (by rule of conditional probability)  

+ $p(C_k \cap (x_1, x_2, ..., x_n)) = p(C_k, x_1, x_2, ..., x_n) = p(x_1, x_2, ..., x_n, C_k)$
  + $= p(x_1 | x_2, ..., x_n, C_k) \times p(x_2, ..., x_n, C_k)$
    + recall that: $P(A \cap B) = P(A, B) = P(A|B) \times P(B)$
  + $= p(x_1 | x_2, ..., x_n, C_k) \cdot p(x_2 | x_3,..., x_n, C_k) \times p(x_3, ..., x_n, C_k)$
  + $= p(x_1 | x_2, ..., x_n, C_k) ... p(x_{n-1} | x_n, C_k) \cdot p(x_n | C_k) \cdot p(C_k) $ -->

## Steps of Naive Bayes Classifier Algorithm

1. Calculate prior probability: $P(C_i)$
2. Calculate conditional probability for each attribute: $P(x_1|C_i), ..., P(x_n|C_i)$
    + if attribute is categorical
      + compute the conditional probability for each category of each attribute
        + i.e. each category of an attribute has one probability
    + if attribute is continuous
      + compute the mean and standard deviation for each attribute
3. Predict on new data
    + assume conditionally independent, i.e. $P(X \mid C_i) = \prod_{k=1}^{n} (x_k \mid C_i)$
      + if categorical
        + using conditional probability from training data to get $P(x_k|C_i)$
      + if continous
        + using mean and standard deviation from training data
          + with Gaussain form: $  g(x_k, \mu, \sigma) = \frac{1}{\sqrt{2 \pi} \cdot \sigma} \times \exp(-\frac{(x - \mu)^2}{2 \sigma^2})$
        + $P(x_k | C_i) = g(x_k, \mu_{C_i}, \sigma_{C_i})$
    + choose the class with maximum posterior probability by $P(C_i) \times P(X \mid C_i)$

## Backgrounds of Naive Bayes Classifier

### Bayes' theorem

[Bayes' theorem](https://en.wikipedia.org/wiki/Bayes%27_theorem)  
In probability theory and statistics, Bayes' theorem (alternatively Bayes' law or Bayes' rule) describes the probability of an event, based on prior knowledge of conditions that might be related to the event.  

$ P(A|B) = \frac{P(B|A) \times P(A)}{P(B)}$  

+ where $A$ and $B$ are events and $P(A) \ne 0$
  + $P(A|B)$ and $P(B|A)$ are both conditional probability
    + former is the 'likelihood' of event $A$ occurring given $B$ is true
    + later is vice versa

### Conditional probability

[Conditional probability](https://en.wikipedia.org/wiki/Conditional_probability)  
If the event of interest is A and the event B is known or assumed to have occurred, "the conditional probability of A given B", or "the probability of A under the condition B", is usually written as $P(A|B)$, or sometimes as $P(A/B)$.  

If $P(A|B) = P(A)$, then events A and B are said to be independent: in such a case, having knowledge about either event does not change our knowledge about the other event.  

+ $P(A \mid B) = \frac{P(A \cap B)}{P(B)}$
  + where $P(A \cap B)$ is the probability that both events A and B occur
+ $P(A \cap B) = P(A, B) = P(A \mid B) \times P(B) $

#### Posterior Probability, 後驗機率

[Posterior probability](https://en.wikipedia.org/wiki/Posterior_probability)  
&nbsp;&nbsp;&nbsp;&nbsp; In Bayesian statistics, the posterior probability of a random event or an uncertain proposition is the conditional probability that is assigned after the relevant evidence or background is taken into account.  
&nbsp;&nbsp;&nbsp;&nbsp; 在貝葉斯統計中，**一個隨機事件或者一個不確定事件的後驗機率是在考慮和給出相關證據或數據後所得到的條件機率**。

#### Prior Probability, 先驗機率

[Prior probability](https://en.wikipedia.org/wiki/Prior_probability)  
In Bayesian statistical inference, a prior probability distribution, often simply called the prior, of an uncertain quantity is the probability distribution that would express one's beliefs about this quantity before some evidence is taken into account.

### Conditional Independence

[Conditional independence, wikipedia](https://en.wikipedia.org/wiki/Conditional_independence)  
&nbsp;&nbsp;&nbsp;&nbsp; Two random events A and B are conditionally independent [given the third event Y] if and only if, given knowledge of whether Y occurs, knowledge of whether A occurs provides _no_ information on the likelihood of B occurring, and knowledge of whether B occurs provides no information on the likelihood of A occurring.  

i.e. $(A \perp B) \mid Y \iff \Pr(A \mid B \cap Y) = \Pr(A \mid B, Y)=\Pr(A \mid Y)$

+ Recall that, two events $A$ and $B$ are **independent**, if
  + $P(A \cap B) = P(A, B) = P(A) \times P(B)$
+ Two events $A$ and $B$ are **conditionally independent**, given event $C$ if
  + $P(A, B \mid C) = P \big((A \cap B) \mid C \big) = P(A \mid C) \times P(B \mid C)$
    + $P(A, B \mid C)$ to denote joint distribution of $A$ and $B$ conditioned on $C$
  + $P(A \mid B, C) = P(A \mid C)$, proof of the equation:

#### Prove $P(A, B \mid C) = P(A \mid C) \times P(B \mid C)$

1. $P(A, B \mid C) = P\big((A, B) \mid C\big)$
2. $= \frac{P\big(A, (B, C)\big)}{P(C)} = \frac{P\big(A \mid (B, C)\big) \times P(B, C)}{P(C)} = P\big(A \mid (B, C)\big) \times \frac{P(B, C)}{P(C)}$
3. $= P\big(A \mid (B, C)\big) \times P(B \mid C)$, (by conditional independence)
4. $= P(A \mid C) \times P(B \mid C)$

#### Prove $P(A \mid B, C) = P(A \mid C)$

1. $P(A \mid B, C) = P\big(A \mid (B, C) \big)$
2. $= \frac{P\big(A, (B, C)\big)}{P(B, C)} = \frac{P\big((A, B), C \big)}{P(B, C)} = \frac{P(A, B \mid C) \times P(C)}{P(B, C)}$, (by conditional probability)
3. $= \frac{P(A \mid C) \times P(B \mid C) \times P(C)}{P(B, C)}$, (by conditional independence)
4. $= \frac{P(A \mid C) \times \frac{P(B, C)}{P(C)} \times P(C)}{P(B, C)} = P(A \mid C)$, (by conditional probability)

#### Example1 of Conditinoal Independence

+ Two people $A$ and $B$
  + $P(A)$, the probability of person A gets home in time for dinner
  + $P(B)$, the probability of person B gets home in time for dinner
+ Event $Y$, snow storm
  + $P(Y)$, the probability of snow storm hitting the city

##### Both "Independent" and "Conditionally Independent"

Some properties of the two events:  

+ $P(A)$ and $P(B)$ are assumed to be independent
  + i.e. probability of $A$ getting home in time has nothing to do with $B$
+ $P(A|Y)$ and $P(B|Y)$ are also assumed to be independent
  + given snow storm hitting the city, the two probabilities are still independent
  + e.g. knowing the fact that there's snow storm and $A$ NOT getting home in time
    + still knowing nothing about $P(B|Y)$, even with known $Y$
  + $P(A \mid B, Y) = P(A \mid B \cap Y)= P(A \mid Y)$
    + the probability of $A$ given $Y$, is not affected by knowing $B$ or not
    + therefore, $A$ and $B$ are conditionally independent, given $Y$

&nbsp;&nbsp;&nbsp;&nbsp; Conditional independence is the same as normal independence, but _**restricted to the case where you know that a certain condition is or isn't fulfilled**_. Not only can you not find out about $A$ by finding out about $B$ in general (normal independence), but you also can not do so under the condition that there's a snow storm (conditional independence).

##### "Independent" but "NOT Conditionally Independent"

i.e. Independent but Conditionally Dependent  

+ Two people $A$ and $B$
  + _$A$ and $B$ lives in the same neighborhood_, i.e. traffic condition is similar

Some properties of the two events:  

+ $P(A)$ and $P(B)$ are assumed to be independent
+ $P(A|Y)$ and $P(B|Y)$ are _NOT independent_
  + e.g. knowing the fact that there's snow storm and $A$ NOT getting home in time
    + we gain information about $P(B|Y)$ by knowing P(A|Y)

#### Example2 of Conditinoal Independence

+ Two dice: $B$ for Blue die and $R$ for Red die
  + the rolling result of each die is independent
  + _$P(B)$ and $P(R)$ are independent of each other_

##### Conditionally Independent

+ After roll on each die, given $info$ (information) that:
  + roll of $B$ is "not 6"
  + roll of $R$ is "not 1"
+ The two events conditional on $info$ are still independent
  + $P(B \mid info)$ and $P(R \mid info)$ are independent
  + $P(B \mid R, info) = P(B \mid info)$

By taking a look at the blue die, we can't gain any knowledge about the red die; after looking at the blue die we will still have a probability of 1/5 for each number on the red die except 1. So the probabilities for the results are conditionally independent given the information given.  

##### NOT Conditionally Independent

+ After roll on each die, given $info$ (information) that:
  + sum of the two results is $even$
  + this allows new information on $R$ to be learned by knowing the result of $B$

For instance, if we see a 3 on the blue die, the red die can only be 1, 3 or 5. So in this case the probabilities for the results are not conditionally independent given this other information.  

&nbsp;&nbsp;&nbsp;&nbsp; Note that **conditional independence is always relative to the given condition** -- in this case, the results of the dice rolls are conditionally independent with respect to the event "the blue result is not 6 and the red result is not 1", but they're not conditionally independent with respect to the event "the sum of the results is even".  

### Other Navie Bayes algorithms

Gaussian: It is used in classification and it assumes that features follow a normal distribution.

Multinomial: It is used for discrete counts. For example, let’s say,  we have a text classification problem. Here we can consider bernoulli trials which is one step further and instead of “word occurring in the document”, we have “count how often word occurs in the document”, you can think of it as “number of times outcome number x_i is observed over the n trials”.

Bernoulli: The binomial model is useful if your feature vectors are binary (i.e. zeros and ones). One application would be text classification with ‘bag of words’ model where the 1s & 0s are “word occurs in the document” and “word does not occur in the document” respectively.

## Appendix: References

General:  

+ [A practical explanation of a Naive Bayes classifier](https://monkeylearn.com/blog/practical-explanation-naive-bayes-classifier/)

General + Code:  

+ [Understanding Naive Bayes Classifier from scratch : Python code](https://appliedmachinelearning.blog/2017/05/23/understanding-naive-bayes-classifier-from-scratch-python-code/)
+ [Naive Bayes Classifier From Scratch](https://chrisalbon.com/machine_learning/naive_bayes/naive_bayes_classifier_from_scratch/)
+ [How To Implement Naive Bayes From Scratch in Python](https://gist.github.com/wzyuliyang/883bb84e88500e32b833)
+ [Naïve Bayes from Scratch using Python only— No Fancy Frameworks](https://towardsdatascience.com/na%C3%AFve-bayes-from-scratch-using-python-only-no-fancy-frameworks-a1904b37222d)
+ [Implementing a Multinomial Naive Bayes Classifier from Scratch with Python](https://medium.com/@johnm.kovachi/implementing-a-multinomial-naive-bayes-classifier-from-scratch-with-python-e70de6a3b92e)
+ [Naive Bayes Classifiers](https://www.geeksforgeeks.org/naive-bayes-classifiers/)

Conditional Independence:  

+ [Could someone explain conditional independence?](https://math.stackexchange.com/questions/23093/could-someone-explain-conditional-independence)
+ [Proving conditional independence](https://math.stackexchange.com/questions/832370/proving-conditional-independence)
+ [Understanding the chain rule in probability theory](https://math.stackexchange.com/questions/228811/understanding-the-chain-rule-in-probability-theory)
+ [Chain rule (probability)](https://en.wikipedia.org/wiki/Chain_rule_(probability))
