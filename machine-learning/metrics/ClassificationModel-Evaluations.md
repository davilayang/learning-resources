
# Understand Precision and Recall

## The Example

&nbsp;&nbsp;&nbsp;&nbsp; In the project of CS910, I modelled a binary classifier to classify foods as healthy or unhealthy, define an attribute `is_healthy` with binary value: _1 for being healthy_, _0 for being unhealthy_. The testset has 14622 observations.  

&nbsp;&nbsp;&nbsp;&nbsp; With KNN, the model has 93.80% accuracy, and the classification result is:  

1. Correctly classify "Healthy" products as "Healthy":
    + i.e. label healthy (real) product as healthy
    + i.e. **True Positive, TP**
    + _Positive tuples correctly labelled as Positive_ (p.402)
    + 10280 rows
2. Incorrectly classify "Unhealthy" products as "Healthy":
    + i.e. label unhealthy (real) product as healthy
    + i.e. **False Positive, FP**
    + _Negative tuples incorrectly labelled as Positive_
    + 212 rows
3. Incorrectly classify "Healthy" products as "Unhealthy":
    + i.e. label healthy (real) product as "unhealthy"
    + i.e. **False Negative, FN**
    + _Positive tuples incorrectly labelled as Negative_
    + 696 rows
4. Correctly classify "Unhealty" products as "Unhealthy":
    + i.e. label unhealthy (real) product as "unhealthy"
    + i.e.**True Negative, TN**
    + _Negative tuples correctly labelled as Negative_
    + 3434 rows

## Confusion Matrix

By definition on wikipedia [wiki_1](https://en.wikipedia.org/wiki/Confusion_matrix) [wiki_2](https://zh.wikipedia.org/wiki/ROC%E6%9B%B2%E7%BA%BF), the matrix structure should be:

|                        |  True (Real)   |  False (Real)  |
|:----------------------:|:--------------:|:--------------:|
|  **True (Predicted)**  | True Positive  | False Positive |
| **False (Predicted)**  | False Negative | True Negative  |

Or, in short term:

|            |       $T$      |       $F$      |
|:----------:|:--------------:|:--------------:|
|  **$T'$**  | True Positive  | False Positive |
|  **$F'$**  | False Negative | True Negative  |

But in `sklearn.metrics.confusion_matrix`, there're huge differences, and it's really "confusing". Most notablly, **the predicted values in columns, real values are in rows**, and default label order is by $[0, 1]$, i.e. $[False, True]$. 

```python
confusion_matrix(y_true=y_test, y_pred=knn_prd)
# same as:
# confusion_matrix(y_true=y_test, y_pred=knn_prd, labels = [0, 1])
# confusion_matrix(y_true=y_test, y_pred=knn_prd, labels = [False, True])
```

returns a confusion matrix as:

|           |      $F'$      |      $T'$      |
|:---------:|:--------------:|:--------------:|
|  **$F$**  | True Negative  | False Positive |
|  **$T$**  | False Negative | True Negative  |

```python
confusion_matrix(y_true=y_test, y_pred=knn_prd, labels=[1, 0])
# confusion_matrix(y_true=y_test, y_pred=knn_prd, labels=[True, False])
```

returns a confusion matrix as:

|           |      $T'$      |      $F'$      |
|:---------:|:--------------:|:--------------:|
|  **$T$**  | True Positive  | False Negative |
|  **$F$**  | False Positive | True Negative  |

```python
confusion_matrix(y_true=y_test, y_pred=knn_prd, labels=[1, 0]).transpose()
```

using `pandas.DataFrame.transpose`, we can get a confusion matrix as by definition in wikipedia.

|            |       $T$      |       $F$      |
|:----------:|:--------------:|:--------------:|
|  **$T'$**  | True Positive  | False Positive |
|  **$F'$**  | False Negative | True Negative  |

## Precision, Recall, and other measures

### 1. Precision

Also called **Positive Predictive Value (PPV)**.
$Precision = PPV = \cfrac{TruePositive}{TruePositive + FalsePositive}$

&nbsp;&nbsp;&nbsp;&nbsp; Intuitively, it's **how good a classfier is at NOT to lable product as healthy when it's actually belonged to unhealty**. Precision can be thought of as a _measure of **exactness**_, i.e., what percentage of tuples labeled as positive are actually such. The value will be higher, if:

+ $TruePositive$ in nominator in increased to _larger_ values
  + $TP$ = label healthy (real) product as healthy
+ $FalsePositive$ in denominator is reduced to _smaller_ values
  + $FP$ = label unhealthy (real) product as "healthy"
  + To minimize the wrongful classification on negative cases

Easier way to remember: _how precision you are, at telling a true negative case is negative, instead of calling it positive_.  

### 2. Recall/ Sensitivity

Also called **Sensitivity**, **Hit Rate**, or **True Positive Rate (TPR)**.  
$Recall = TPR = \cfrac{TruePositive}{TruePositive + False Negative} = 1 - FNR$  

&nbsp;&nbsp;&nbsp;&nbsp; Intuitively, it's **how good a classifier is at finding all positive products**. Recall can be thought of as a _measure of **completeness**_, i.e. what percentage of positive tuples are labeled as such. The value will be higher, if:

+ $TruePositive$ in nominator in increased to _larger_ values
  + $TP$ = label healthy (real) product as healthy
+ $FalseNegative$ in denominator is reduced to _smaller_ values
  + $FN$ = label healthy (real) product as "unhealthy"
  + To minimize the wrongful classifications on positive cases

Easier way to remember: _recall from memory, the positive cases, but you may recall wrong: negative cases as positive_.  

### 3. Specificity

Also called **Selectivity** or **True Negative Rate (TNR)**.  
$Specificity = TNR = \cfrac{TrueNegative}{TrueNegative + FalsePositive} = 1 - FPR$  

&nbsp;&nbsp;&nbsp;&nbsp; Intuitively, it's **how good a classfier is at finding all negative samples**. The value will be higher, if:

+ $TrueNegative$ in nominator in increased to _larger_ values
  + $TN$ = label umhealthy (real) product as unhealthy
+ $FalsePositive$ in denominator is reduced to _smaller_ values
  + $FP$ = label unhealthy (real) product as healthy
  + To minimize the wrongful classifications on negative cases
  + Precision and Specificiy both have FP in denominator

### 4. $F$-Score/ $F_1$ score

Harmonic mean of precision and recall:  
$F_1$ score = $\cfrac{2 \times precision \times recall}{precision + recall}$

Weighted measure of precision and recall:  
$F_\beta$ score = $\cfrac{(1 + \beta^2) \times precision \times recall}{\beta^2 \times precision + recall}$, where $\beta$ is a non-negative real number

### 5. Receiver Operating Characteristics (ROC) Curve

&nbsp;&nbsp;&nbsp;&nbsp; The ROC curve is created by plotting the true positive rate (TPR) against the false positive rate (FPR = 1 - Specificity) _at various threshold settings_ [wiki_3](https://en.wikipedia.org/wiki/Receiver_operating_characteristic).  
&nbsp;&nbsp;&nbsp;&nbsp; ROC curves typically feature true positive rate on the Y axis, and false positive rate on the X axis. **This means that the top left corner of the plot is the “ideal” point - a false positive rate of zero (FPR = 0), and a true positive rate of one (TPR = 1)**. This is not very realistic, but it does mean that a larger area under the curve (AUC) is usually better.  

<!-- ![ROC Figure](.\figures\fig11_aucroc.png) -->
<p align="center"> <img src='.\figures\fig11_aucroc.png' width='50%'> </p>

## Making Plots with Python

### How to plot AUC-ROC Curve

1. Traing a model of classificatoin
2. Get the scores on testset, depending on the model used, can be:  
    (i.e. the model must be able to return a probability of the predictec class for each test tuple)  
    (example of `sklearn.linear_model.LogisitcRegression`)
    + `.predict_proba(X_test)`
    + `.predict_log_proba(X_test)`
    + `.decision_function(X_test)`
3. Get the scores for positive cases
    + if `.predict_proba()` or `.predict_log_proba()`
        + `.predict_proba(X_test)[:, 1]`
    + if `.decision_function()`
        + `.decision_function(X_test)`
4. Use `sklearn.metrics.roc_curve` to compute ROC scores
    + given y_test, labels for testset
    + given y_score, can either be:
        + _probability estimates of the positive class_
        + _confidence values_
        + _non-thresholded measure of decisions (returned by “decision\_function”)_
    + Note that number of threshold is decided by kind of y_score feeded
        + with KNN or Decision Tree, predicted score is either 0 or 1, not many thresholds
        + with Logistic Regression, it can be probability from 0 to 1, many thresholds
5. Use `sklearn.metrics.auc` to compute the area under ROC curve
    + given array of FPR and TPR
6. plot by `seaborn.lineplot` or `matplotlib.pyplot.plot`

Example codes:

```python
from sklearn.linear_model import LogisitcRegression
from sklearn.metrics import roc_curve, auc
# X_train, X_test, y_train, y_test = train_test_split()
# lr = LogisitcRegression()
# Assume training is done
test_probs = lr.predict_log_proba(X_test)
y_score = test_probs[:, 1]
# compute ROC scores and AUC
lr_fpr, lr_tpr, threshold = roc_curve(y_test, y_score)
lr_roc_auc = auc(lr_fpr, lr_tpr)
# plot AUC ROC
plt.plot(lr_fpr, lr_tpr, label='AUC ROC = %0.2f' % lr_roc_auc)
plt.show()
```

### How to plot Precision-Recall Curve

```python
clf_pred_probs = clf.predict_proba(X=binary_class_testing_data)
precision, recall, _ = metrics.\
    precision_recall_curve(y_true=binary_class_testing_labels, \
                           probas_pred=clf_pred_probs[: ,1], pos_label=9)
plt.step(recall, precision, color=clrs[idx], where='post')
plt.fill_between(recall, precision, color=clrs[idx], alpha=0.5, step='post')
```

### How to plot subplots with `for-loop`

```python
fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
fig.subplots_adjust(top=0.93, hspace=.15, wspace=.05)
axes = axes.ravel()

for idx in range(4):
    # plot for each subplot
    axes[idx].plot(x=data_x, y=data_y)
    # setting for each subplot
    axes[idx].set_title('Subplot Title')
    axes[idx].set_xlabel('Subplot X-axis label')
    axes[idx].set_ylabel('Subplot Y-axis label')
    axes[idx].legend(loc='lower right', fontsize=12)

# setting for the main plot
plt.suptitle('Plot Main Title', fontsize=20)
plt.show()
```

+ [https://matplotlib.org/api/axes_api.html](https://matplotlib.org/api/axes_api.html)
+ [https://matplotlib.org/api/axes_api.html#plotting](https://matplotlib.org/api/axes_api.html#plotting)


## References



<!-- ### AUC (Area Under The Curve) -->