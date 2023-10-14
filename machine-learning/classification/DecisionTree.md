# About Decision Tree

<!-- markdownlint-disable MD033 -->

<style>
p { text-indent: 5%; }
li { margin-left: -15px; }
color1 { color: crimson; }
</style>

## Intuitions of Decision Tree

- [Decision tree learning, wikipedia](https://en.wikipedia.org/wiki/Decision_tree_learning)  
- [A Complete Tutorial on Tree Based Modeling from Scratch (in R & Python)](https://www.analyticsvidhya.com/blog/2016/04/complete-tutorial-tree-based-modeling-scratch-in-python/#one)  

Decision tree is a type of supervised learning algorithm (having a pre-defined target variable) that is mostly used in classification problems. It works for both categorical and continuous input and output variables. In this technique, we **split the population or sample into two or more homogeneous sets (or sub-populations) based on most significant splitter / differentiator in input variables**.  

Some techniques, often called ensemble methods, construct more than one decision tree:

- <color1>**Boosted** trees</color1>
  - **Incrementally building an ensemble by training each new instance to emphasize the training instances previously mis-modelled**
  - e.g. AdaBoost
- <color1>**Bootstrap aggregated (or bagged)** decision trees</color1>
  - **Builds multiple decision trees by repeatedly resampling training data with replacement, and voting the trees for a consensus prediction**
- **Rotation** forest
  - every decision tree is trained by first applying principal component analysis (PCA) on a random subset of the input features.

### Terminology related to Decision Trees

- **Root Node**:
  - represents entire population or sample
  - further gets divided into two or more _homogeneous sets_
- **Splitting**:
  - a process of dividing a node into two or more sub-nodes
- **Decision Node**:
  - when a sub-node splits into further sub-nodes, it is called decision node
- **Leaf/ Terminal Node**:
  - nodes that do not split is called Leaf or Terminal node.
- **Pruning**:
  - when we remove sub-nodes of a decision node, this process is called pruning
  - it's the opposite process of splitting
- **Branch / Sub-Tree**:
  - a sub section of entire tree is called branch or sub-tree
- **Parent and Child Node**:
  - a node, which is divided into sub-nodes is called parent node of the sub-nodes
  - those sub-nodes are the child of parent node

### Advantages

1. Easy to understand or interpret
2. Useful in data exploration
   - one of the fastest way to identify most significant variables
   - e.g. with hundreds of variables, decision tree will help to identify most significant ones
3. Less data cleaning required
4. Data type is not a constraint: handle both numerical and categorical
5. Non Parametric Method
   - decision trees have no assumptions about the space distribution and the classifier structure

### Disadvantages

1. Problem of Over-fitting
   - can be solved by setting constraints on model parameters and pruning
2. Not fit for continuous variables
   - decision tree loses information when it categorizes variables into different categories (ranges)

## Steps of Decision Tree Algorithm

1. Start with all observations at "root"
2. Choose an attribute to split on, and choose a split value
   - Different algorithms use different metrics for splitting
     - **C4.5, using Information Gain**
       - Attribute with higher gain is selected
     - **CART, using Gini Impurity**
   - if attribute is categorical, split value use the categories
   - if attribute is continuous, values are discretized (e.g. mean or median) then split
3. Recursively apply (1) and (2) on both halves of "root"
4. Tree split is done until _stop condition_ is met:
   - All observations in a node belong to the same class
   - All observations in a node have the same attribute values
   - Only one observation in a node
5. Label each 'leaf' with the majority class

## Backgrounds of Decision Tree

1. About Tree Metrics
2. C4.5 Algorithm
3. CART & Gini Impurity
4. Information Gain
5. Other Metrics for Tree Split
6. Tree Pruning

### Tree Metrics

Different algorithms use different metrics for measuring "best" split. These generally **measure the homogeneity of the target variable within the subsets**. These metrics are applied to each candidate subset, and the resulting values are combined (e.g., averaged) to provide a measure of the quality of the split.  

For any decision tree algorithm, you need **a measure of “inequality” or “uniformity”** that acts as a heuristic for choosing “good” splits at each level in the tree. "Good splits" separate categorical data into child nodes with small Gini impurities (or large Information Gain), because this _indicates an unequal distribution of labels_.  

### C4.5 algorithm

[C4.5 algorithm, wikipedia](https://en.wikipedia.org/wiki/C4.5_algorithm)  

### Gini Impurity/ Gini Index

[Gini impurity, wikipedia](https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity) (not the same as "Gini Coefficient")  

<color1>Gini impurity (sometimes called "Gini Index") is a measure of **how often a randomly chosen element from the set would be incorrectly labelled** if it was randomly labelled according to the distribution of labels in the subset</color1>.

<color1>Gini Impurity is a measurement of **the likelihood of an incorrect classification of a new instance of a random variable**, if that new instance were **randomly classified according to the distribution of class labels** from the data set</color1>.

The Gini impurity can be computed by _"summing the probability $p_i$ of an item with class/label $i$ being chosen" times "the probability of a mistake in categorizing that item"_.  

$$ \text{Gini Impurity} = I_{Gini}(T) = Gini(T) = 1 - \sum_{c \in C} ({p_c})^{2}$$

- where $p_c$ is the probability of class/label $c$ being selected
  - $T$ for a set of training examples
  - $C$ for the set of classes/labels, e.g. if binary, $C = \{\text{positive, negative}\}$
- To derive the formula:
  - $\operatorname I_{Gini}(p) = \sum\limits_{c \in C} \big(p_c \times \sum\limits_{k \neq c} p_k \big)$
  - $= \sum\limits_{c \in C} \big( p_c \times (1-p_c) \big) =  \sum\limits_{c \in C} (p_c-{p_c}^2) =  \sum\limits_{c \in C} p_c -  \sum\limits_{c \in C} {p_c}^{2} = 1 - \sum\limits_{c \in C} {p_c}^{2}$  
    (sum of the probability of each class is 1)  
  - $k$ for the "wrong class" of an data/item, therefore $c \ne k$
    - e.g $C = \{1, 2, 3\}$ and item has class $1$, i.e. $k \in \{2, 3\}$
    - sometimes proability is referred as proportion

Gini Impurity **reaches its minimum (zero) when all cases in the node fall into a single target category**, i.e. if there's only one class, gini impurity is zero, since there'll be no incorrect classifications (there's no impurity).  

#### Steps for calculating Gini Impurity and Tree-split

1. Discretize the attribute if necessary, use the result for split
   - if attribute type is _continuous_, discretize it by a value
     - can be mean, median...etc
       - e.g. if from $4.0$ to $6.0$, discretize by $\ge 5$ and $< 5$ (if binary split)
   - if attribute type is _categorical_, use the categories
2. Calculate weighted Gini Impurity/Index for each attribute
   1. compute Gini Impurity of split subsets on an attribute
      - $Gini(S_a(v))$, for impurity of a subset on attribute $a$
        - $v$ for the split condition, data in subset $S_a(v)$ must meet the split condition
        - e.g. split by $5$, compute impurity of subset $\ge 5$ and subset $< 5$
   2. weight and sum the results to get attribute Gini Impurity $Gini(T|a)$
      - $Gini(T | a) = \sum \cfrac{|S_a(v)|}{|T|} \cdot Gini(S_a(v))$
        - $|S_a(v)|$ for the number of observation in a split subset
        - $|T|$ for the number of observation in dataset $T$
3. Use the attribute with _lowest_ Gini Impurity at 'root'
   - i.e. use _the purest_ attribute
     - to split into subsets that having unequal distribution of labels
     - our split final goal is to make the node have only one class
   - at next split, perform 1. ~ 2. with the subset, recursively
4. Tree is constructed when stop condition is met

#### Interpretation of Gini Impurity

- <color1>if $Gini(T) = 0$</color1>
  - there's only one class in the dataset, i.e. no chance for incorrect classifications
  - or we can say the node is _purest_.
  - or there's no chance for 'mis-classification', since no other labels to select
- <color1>if $1 > Gini(T) > 0$</color1>
  - **Based on the observed training data,**
    - **new input data has $Gini(T)$ chance to be incrrectly classified**
  - with higher impurity, there're more options/possibilties for 'mis-classifications'

### Information Gain

[Information gain in decision trees, wikipedia](https://en.wikipedia.org/wiki/Information_gain_in_decision_trees)  

By using information gain as a criterion, we **try to estimate the information contained by each attribute**. Ideally, the attribute with the highest information gain is the attribute to split, i.e. _among the attributes, this attribute has the most information, therefore we split on it_.  

<color1>In general terms, the expected information gain is **the change in information entropy $Η$ from a prior state to a state that takes some information** as given</color1>:  

$$ \text{Information Gain} = IG(T|a) = H(T) - H(T|a)$$  

- where $H(T|a)$ is the conditional entropy of $T$ given the value of attribute $a$
  - $T$ for a set of training examples, each of the form $(x, y)$
  - $a$ for one of the attributes in the dataset

#### Steps for calculating Information Gain and Tree-split

1. Calculate entropy of the class/label/target
   - $H(T) = - \sum\limits_{c \in C}^{} \big( p_{c} \times \log_2 (p_{c}) \big)$
     - $T$ for the dataset, with form $(x, y) = \big( (x_1, x_2, x_3, ...), y \big)$
     - $C$ for each class/label for the dataset
   - e.g. if $class \in \{\text{positive, negative}\}$, compute entropy for the two class
2. Discretize the attribute if necessary, use the result for split
3. Calculate weighted conditional entropy for every attribute
   1. compute entropy of split subsets on an attribute
      - $H(S_a(v))$, for entropy of a subset on attribute $a$
        - $v$ for the split condition, data in subset must meet the split condition
      - e.g. split by $5$, compute entropy of subset $\ge 5$ and subset $< 5$
   2. weight and sum the results to get $H(T|a)$, i.e. entropy given attribute $a$
      - $H(T | a) = \sum \cfrac{|S_a(v)|}{|T|} \cdot H(S_a(v))$
        - $|S_a(v)|$ for the number of observations in a split  subset
        - $|T|$ for the number of observations in dataset $T$
4. Information Gain $=$ $H$ from prior state $-$ $H$ after given some information (of $a$)
   - $H(T) - H(T|a)$
5. Use the attribute with _highest_ information gain at 'root'
    - at next split, perform 1. ~ 4. with the subset, recursively
6. Tree is constructed when stop condition is met

#### Information Content

[Information content, wikipedia](https://en.wikipedia.org/wiki/Information_content)  

In information theory, information content, self-information, or **surprisal of a random variable** or signal is **the amount of information gained when it is sampled**. Formally, **information content is a random variable defined for any event in probability theory** regardless of whether a random variable is being measured or not.  

The _expected value_ of self-information is **information theoretic entropy**, **the average amount of information an observer would expect to gain about a system when sampling the random variable**.

### Other Metrics for Tree Split

### Tree Pruning

[Decision tree pruning](https://en.wikipedia.org/wiki/Decision_tree_pruning)  
Pruning is a technique in machine learning that reduces the size of decision trees by removing sections of the tree that provide little power to classify instances. Pruning reduces the complexity of the final classifier, and hence improves predictive accuracy by the reduction of over-fitting.  

#### Pre-Pruning

A tree is pruned by halting its construction early (i.e. , by deciding not to further split or partition the subset of training tuples at a given node). Measures such as statistical significance, information gain, Gini index, can be used to assess the _goodness of a split_. If partitioning the tuples at a node would result in a split that falls below a pre-specified threshold, then further partitioning of the given subset is halted.  

#### Post-Pruning

Another common approach is removing subtrees from a 'fully grown' tree.  A subtree at a given node is pruned by removing its branches and replacing it with a leaf.  

##### Reduced error pruning

Start from bottom of fully grown tree:  

- Replace node with majority class
- Measure errors using cross-validation set
  - if accuracy is not affected
    - accept
  - if accuracy is reduced
    - stop

##### Cost complexity pruning

[Cost-Complexity Pruning](http://mlwiki.org/index.php/Cost-Complexity_Pruning)  

## Appendix: References

General:  

- [Tree algorithms: ID3, C4.5, C5.0 and CART, sklearn](https://scikit-learn.org/stable/modules/tree.html#tree-algorithms-id3-c4-5-c5-0-and-cart)
- [scikit-learn : Decision Tree Learning I - Entropy, Gini, and Information Gain](https://www.bogotobogo.com/python/scikit-learn/scikt_machine_learning_Decision_Tree_Learning_Informatioin_Gain_IG_Impurity_Entropy_Gini_Classification_Error.php)
- [A Complete Tutorial on Tree Based Modeling from Scratch (in R & Python)](https://www.analyticsvidhya.com/blog/2016/04/complete-tutorial-tree-based-modeling-scratch-in-python/)
- [A Step By Step C4.5 Decision Tree Example](https://sefiks.com/2018/05/13/a-step-by-step-c4-5-decision-tree-example/)
- [Chapter 4: Decision Trees Algorithms](https://medium.com/deep-math-machine-learning-ai/chapter-4-decision-trees-algorithms-b93975f7a1f1)
- [Decision Tree Introduction with example](https://www.geeksforgeeks.org/decision-tree-introduction-example/)
- [Introduction to Decision Tree Algorithm](http://dataaspirant.com/2017/01/30/how-decision-tree-algorithm-works/)

General - Codes:  

- [Pure Python Decision Trees](http://kldavenport.com/pure-python-decision-trees/)
- [Decision Tree from the Scratch](https://medium.com/@rakendd/decision-tree-from-scratch-9e23bcfb4928)
- [Random forests and decision trees from scratch in python](https://towardsdatascience.com/random-forests-and-decision-trees-from-scratch-in-python-3e4fa5ae4249)
- [How To Implement The Decision Tree Algorithm From Scratch In Python](https://machinelearningmastery.com/implement-decision-tree-algorithm-scratch-python/)
- [Decision Trees and Political Party Classification](https://jeremykun.com/2012/10/08/decision-trees-and-political-party-classification/)

Gini Impurity:  

- [What is the interpretation and intuitive explanation of Gini impurity in decision trees?](https://www.quora.com/What-is-the-interpretation-and-intuitive-explanation-of-Gini-impurity-in-decision-trees)
- [Gini Impurity (With Examples)](https://bambielli.com/til/2017-10-29-gini-impurity/)
- [Interpretation of the Gini impurity](https://pixorblog.wordpress.com/2017/11/06/interpretation-of-the-gini-impurity/)
- [A simple & clear explanation of the Gini impurity?](https://stats.stackexchange.com/questions/308885/a-simple-clear-explanation-of-the-gini-impurity/308886)

Information Gain:  

- [entropy, dictionary.com](https://www.vocabulary.com/dictionary/entropy)
- [Entropy (Information Theory)](https://brilliant.org/wiki/entropy-information-theory/)
- [Entropy in information theory](http://settheory.net/information-entropy)
- [[ML] Decision Tree rule selection: Information Gain v.s. Gini Impurity](http://haohanw.blogspot.com/2014/08/ml-decision-tree-rule-selection.html)
- [Information Entropy and Information Gain](https://bambielli.com/til/2017-10-22-information-gain/)
