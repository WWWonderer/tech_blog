---
title: "ML Basics - Naive Bayes Classification Revisited"
date: 2026-01-01
categories: "machine_learning"
---

$$
\begin{aligned}
&\text{Given data:} 
&&X \in \mathbb{R}^{n \times d}, \; y \in \{1,\dots,K\}^{n \times 1} \\

&\text{Prediction (for } x \in \mathbb{R}^d\text{):} 
&&\widehat{y} = \arg\max_k \;
\left[
\log P(y=k) + \sum_{i=1}^d \log P(x_i \mid y=k)
\right]
\end{aligned}
$$

**Generative vs discriminative model**

In contrast to the linear classifiers discussed in the previous post, Naive Bayes does not explicitly model a decision boundary. Instead, it models the class-conditional distributions of the data and uses these distributions to perform inference on new inputs. Parameter estimation in Naive Bayes is purely analytical, relying on elementary statistical estimates rather than iterative optimization or approximation-based training. Such a setup offers a degree of interpretability, since predictions are driven by explicit class priors and feature-level statistics.

**Underlying principle**

The underlying principle of Naive Bayes is simple: assuming that features are conditionally independent given the class, the joint likelihood of the observed features under each class provides a useful signal for classification. More precisely, for a single data point $x = \\{x_1, x_2, \dots, x_d\\}$ with $d$ features and $K$ possible class labels, Naive Bayes assumes conditional independence of features given the class (the naive part). That is, for any class $k$,

$$
P(x \mid y = k) = \prod_{i=1}^d P(x_i \mid y = k).
$$

Then, by Bayes' rule,

$$
\begin{aligned}
P(y = k \mid x) 
&= \frac{P(x \mid y = k)\, P(y = k)}{P(x)} \\
&= \frac{\left(\prod_{i=1}^d P(x_i \mid y = k)\right) P(y = k)}{P(x)}.
\end{aligned}
$$

Since $P(x)$ does not depend on $k$, it can be ignored when selecting the most likely class:

$$
\begin{aligned}
\widehat{y} 
&= \arg\max_k \; P(y = k \mid x) \\
&= \arg\max_k \; \frac{P(y = k)\prod_{i=1}^d P(x_i \mid y = k)}{P(x)} \\
&= \arg\max_k \; P(y = k)\prod_{i=1}^d P(x_i \mid y = k).
\end{aligned}
$$

Now, we can easily and accurately calculate the prior $P(y = k)$ from the $n$ sampled data as so:

$$
P(y=k)=\frac{1}{n}\sum_{i=1}^{n}\mathbf{1}\left\{y_{i}=k\right\}
$$

But it is not straightforward to compute $\prod_{i=1}^{d} P(x_i \mid y = k)$directly from raw data without making assumptions about the form of the class-conditional feature distributions $P(x_i \mid y = k)$. In practice, Naive Bayes relies on simple parametric assumptions, leading to three common variants:

**Bernoulli Naive Bayes**, where each feature is assumed to be binary. For example, in medical diagnosis, symptoms may be represented as present or absent, and counting the same symptom multiple times would not be meaningful. Here, $P(x_i \mid y = k)$ is estimated from data by counting how often $x_i$ is present among samples belonging to class $k$.

**Multinomial Naive Bayes**, where features represent counts. This is commonly used in text classification, where each word corresponds to a feature and its value is the number of times the word appears in a document. Here, $P(x_i \mid y = k)$ is estimated by counting the total number of occurrences of word $x_i$ across all samples belonging to class $k$ (as opposed to the total number of documents in $k$ containing $x_i$ as in the Bernoulli).

**Gaussian Naive Bayes**, where features are assumed to follow a Gaussian distribution conditioned on the class. This is appropriate when features are continuous-valued, such as physical measurements or sensor readings. Here, $P(x_i \mid y = k)$ assumed to follow a Gaussian distribution, with mean and variance estimated from the data of class k.

For numerical stability, we can translate everything into log space:

$$
\begin{aligned}
\widehat{y}
&= \arg\max_k \ P(y = k)\prod_{i=1}^d P(x_i \mid y = k) \\
&= \arg\max_k \left[ \log P(y=k) + \sum_{i=1}^{d} \log P(x_i \mid y=k) \right].
\end{aligned}
$$

Example implementations of the Naive Bayes can be found in the [📓 **Colab notebook**](https://colab.research.google.com/drive/15FGO8N-V8nCJxLwPEjto7JoJ2wp4hZRR?usp=sharing).