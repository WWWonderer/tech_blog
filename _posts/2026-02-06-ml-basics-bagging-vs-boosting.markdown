---
title: "ML Basics - Bagging vs. Boosting"
date: 2026-02-06
categories: "machine_learning"
---

### Ensemble Methods

While simple models such as shallow decision trees have limited predictive power on their own, aggregating many of them (ensemble methods) can produce a strong predictor that, in practice, often rivals more complex models from the deep learning era. Moreover, tree-based ensembles offer a high degree of interpretability, making them particularly attractive in settings where error analysis and model explainability are important. Broadly speaking, there are 2 principle ways to construct such ensembles: bagging and boosting. 

### Bagging (Random Forest)

Perhaps the most intuitive way to construct an ensemble is to train many simple decision trees on different subsets of the original dataset and then aggregate their predictions, typically by averaging (for regression) or majority voting (for classification). This is known as bagging, short for *bootstrap aggregating*.

**Subset construction.** A common way to construct a subset to train individual trees is to perform sampling with replacement on the original data. Specifically, assuming the training set has $$N$$ examples, a bootstrap sample is created by drawing $$N$$ times on the original data using the full data each time. Let $$P(i) = \frac{1}{N}$$ be the probability of drawing the $$i_{th}$$ data, we have:

$$
\begin{aligned}
P(\text{not } i) &= 1 - \frac{1}{N} \\
P(\text{not drawing $i$ in $N$ tries}) &= (1 - \frac{1}{N}) ^ N\\
P(\text{drawing $i$ at least once in $N$ tries})&= 1 - \left(1-\frac{1}{N}\right)^N
\end{aligned}
$$

For a large dataset, we can use the limit on the equation above to estimate the percentage of drawn samples:

$$
\begin{aligned}
\lim_{N \to \infty} \left(1-\frac{1}{N}\right)^N &= e^{-1}\approx 0.37\\
P(\text{drawing $i$ at least once in $N$ tries})&= 1 - \left(1-\frac{1}{N}\right)^N\approx 1 - 0.37 \approx 0.63
\end{aligned}
$$

Therefore, a bootstrap sample of size N contains approximately 63% of the original dataset in expectation, with the remaining 37% left out-of-bag. These out-of-bag examples are useful in ensemble methods because they help decorrelate and distinguish individual trees within the ensemble.

**Aggregation.**
We can then aggregate the results of the individual trees (each trained on its own share of data) by using averaging or majority voting. Specifically, let $\\{ f_1, f_2, ..., f_M \\}$ denote the set of trained trees, where each $f_m(x)$ produces a prediction for input $x$.

For regression, we can simply compute the ensemble prediction $\widehat{f}$ by calculating the average as so:

$$
\widehat{f}(x) = \frac{1}{M} \sum_{m=1}^{M}f_m(x). 
$$

For classfication, each tree predicts a class label $f_m(x) = c$ ($c \in C$), where $C$ is the set representing all the classes. The final prediction of the ensemble would be:

$$
\widehat{f}(x) = \operatorname*{arg\,max}_{c \in C} \sum_{m=1}^{M} \mathbf{1}\{ f_m(x) = c \}
$$

where $\mathbf{1}\lbrace \cdot \rbrace$ is the indicator function. The majority class is thus the ensemble prediction.

### Boosting (Gradient Boosting)

Unlike bagging, where trees are trained independently and then averaged, boosting builds an ensemble sequentially, where each new tree is trained to correct the mistakes of the previous trees. 

**General Formulation.** Specifically, boosting constructs the ensemble prediction as so:

$$
\begin{aligned}
\widehat{f}(x) &= f_0(x) + \sum_{m=1}^M \eta \  h_m(x) \quad \text{or equivalently,} \\
\widehat{f}_{m+1}(x) &= \widehat{f}_{m}(x) + \eta \  h_{m+1}(x)
\end{aligned}
$$

where $f_0(x)$ is the initial predictor, and each $h_m(x)$ is a weak learner trained to correct the mistake of the initial predictor and preceding correction learners. $\eta$ is the learning rate.

**How to quantify and correct mistakes of the previous predictors?** Now, the core of boosting lies on quantifying and correcting mistakes, but it is not obvious how we can do it under a unifying framework. Exactly what is 'wrong' and how 'wrong' are we differs under different evaluation criterias. But J. Friedman found a genius solution by adopting a unifying framework based on a first-order Taylor expansion of the loss function. First, remind that the first-order Taylor expansion takes the following form:

$$
f(x) \approx f(a) + f'(a)(x - a)
$$

where $a$ is a point on the domain of the function $f$. Now, we want to minimize the distance between our current predictor $$\widehat{f}_m(x)$$ to the true labels $y$ using a certain loss function $$L(\widehat{f}_m(x), y)$$, by iteratively adding $\eta h_{m+1}(x)$, resulting in the following optimization problem:

$$
\arg\min_{h} \sum_{i} L(y_i,\, \widehat{f}_{m}(x_i) + \eta \  h_{m+1}(x_i))
$$

Then, we can view the above loss function $L$  as the $f(x)$ in a Taylor-expansion, where we choose the point of expansion $a$ to be the current ensemble prediction $$\widehat{f}_m(x)$$, and the delta $(x - a)$ to be the next weak predictor $\eta \  h_{m+1}(x_i)$. So, for a single observation $i$:

$$
\begin{aligned}
\underbrace{L(y_i, \widehat{f}_m(x_i) + \eta \  h_{m+1}(x_i))}_{f(x)} \approx \underbrace{L(y_i, \widehat{f}_m(x_i))}_{f(a)} + \underbrace{\left[ \frac{\partial L(y_i, \widehat{f}_m(x_i))}{\partial \widehat{f}_m(x_i)} \right]}_{f'(a)} \underbrace{\eta \  h_{m+ 1}(x_i)}_{x - a}
\end{aligned}
$$


As $L(y_i, \widehat{f}_m(x_i))$ is given and unchanged at each timestep, minimizing the loss function is equivalent to minimizing the $f'(a)(x - a)$ part.

Now, when we generalize this to the entire dataset of $N$ datapoints, the above equation becomes: 

$$
\begin{aligned}
L(y, \widehat{f}_m(x) + \eta \  h_{m+1}(x)) \approx \sum_i L(y_i, \widehat{f}_m(x_i)) + \sum_i \left[ \frac{\partial L(y_i, \widehat{f}_m(x_i))}{\partial \widehat{f}_m(x_i)} \right] \eta \  h_{m+ 1}(x_i)
\end{aligned}
$$

notice how now the $f'(a)(x - a)$ part is a dot product of 2 vectors in the **functional space**: $$\widehat{f}(x) = (\widehat{f}_1(x_1), \widehat{f}_2(x_2), ..., \widehat{f}_N(x_N))$$ and $$h_{m+1}(x) = (h_{m+1}(x_1), h_{m+1}(x_2), ..., h_{m+1}(x_N))$$. The dot product of 2 vectors is minimized when they point to the opposite direction of each other ($a \cdot b = \Vert a \Vert \Vert b \Vert cos\theta$, where $cos\theta$ is minimized when $a$ and $b$ are 180 degrees from each other), therefore to most effectively minimize the loss $L$, we set $$h_{m+1} =  -\left[ \frac{\partial L(y_i, \widehat{f}_m(x_i))}{\partial \widehat{f}_m(x_i)} \right]$$.

**Boosting in action.** Now, it is nice (and tedious) to write the above theoretical section, but boosting did not happen because of it. In reality, boosting became popular in the 90s while "no one knew why or how [it worked]"[^1]. A simple example can better illustrate how boosting operates in practice. Consider a simple regression problem with 3 datapoints. We train a sequence of weak learners iteratively, where each new learner is trained to predict the error (or **residual**) of the previous steps. The process is summarized in the table below, assuming a learning rate $\eta$ of $0.1$.

| Iteration $m$ | $x$ | True target $y$ | Current prediction $\hat{f}_m(x)$ | Residual $r_m = y - \hat{f}_m(x)$ | New learner $h_{m+1}(x)$ | Updated model |
|--------------|----|-----------------|------------------------------------|----------------------------------|--------------------------|---------------|
| 1            | 1  | 3               | 1.5                                | 1.5                              | 1.4                      | $\hat{f}_2=\hat{f}_1+\eta h_2$ |
|              | 2  | 2               | 1.5                                | 0.5                              | 0.6                      |               |
|              | 3  | 1               | 1.5                                | −0.5                             | −0.4                     |               |
| 2            | 1  | 3               | 1.64                               | 1.36                             | 1.3                      | $\hat{f}_3=\hat{f}_2+\eta h_3$ |
|              | 2  | 2               | 1.56                               | 0.44                             | 0.4                      |               |
|              | 3  | 1               | 1.46                               | −0.46                            | −0.5                     |               |
| 3            | 1  | 3               | 1.77                               | 1.23                             | 1.1                      | $\hat{f}_4=\hat{f}_3+\eta h_4$ |
|              | 2  | 2               | 1.60                               | 0.40                             | 0.3                      |               |
|              | 3  | 1               | 1.41                               | −0.41                            | −0.4                     |               |
| ...          | |                |                               |                          |                     |               |

Intuitively, as long as each new learner points roughly in the direction of the residuals, the ensemble keeps correcting its own mistakes and gradually improves. Now, we can link this back to the theory above. Notice that if we use the MSE loss, we would have:

$$
\begin{aligned}
\frac{\partial L(y, \widehat{f}_m(x))}{\partial \widehat{f}_m(x)} &= \frac{\partial (y - \widehat{f}_m(x)) ^ 2}{\partial \widehat{f}_m(x)} \\
&= -2 (y - \widehat{f}_m(x))\\
\end{aligned}
$$ 

Now, the $2$ is a constant which can be incorporated into the learning rate $\eta$, and we take a predictor to the opposite direction of the gradient with:

$$
\begin{aligned}
h_{m+1}(x) &= -\frac{\partial L(y, \widehat{f}_m(x))}{\partial \widehat{f}_m(x)} \\
&= 2(y - \widehat{f}_m(x)) \\
&\propto y - \widehat{f}_m(x)
\end{aligned}
$$

where the last equation is exactly the residual in practice.  

### Key Algorithmic Differences Between Bagging and Boosting
In summary, both bagging and boosting are ensemble methods that combine multiple weak predictors to form a stronger model. Bagging trains similar weak learners on different subsets of the original data, and aggregates their predictions through averaging or majority voting. Because these learners are independent, they can be trained in parallel.

Boosting, on the other hand, constructs the ensemble sequentially: each weak learner (except the first) is trained to correct the errors or residuals of the current ensemble. This sequential dependency prevents parallel training, and the optimization proceeds in function space instead of the data space as in bagging. 

### Relevance of Ensemble Methods in the Deep Learning Era
Now, one might ask why are these models still relevant in today's deep learning era? Couldn't we just fit some neural networks which are proven to be universal function approximators that can, in theory, fit anything given enough data and compute? 

First, ensemble methods work surprisingly well in practice. Libraries such as XGBoost frequently achieve comparable or even superior performance to embedding based models in settings such as Kaggle competitions, often with far less complexity and tuning overhead. 

Second, neural networks are prune to overfitting and lack any interpretability. In contrast, ensemble based models inherently reduces variance through aggregation and offer clear explainability to every decision, making them particularly attractive when robustness and explainability are important. 

[^1]: *Interview with Jerry Friedman on Gradient Boosting*, [YouTube, 4:13](https://www.youtube.com/watch?v=ENywgsJoMIA&t=253).
