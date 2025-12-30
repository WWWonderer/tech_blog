---
title: "ML Basics - Linear Classification Revisited"
date: 2025-12-29
categories: "machine_learning"
---

$$
\begin{aligned}
&\text{Given data:} &&X \in \mathbb{R}^{n \times d}, \; y \in \{0,1\}^{n \times 1} \\
&\text{Model parameters:} &&w \in \mathbb{R}^{d \times 1} \\
&\text{Linear score:} &&z = Xw \\
&\text{Prediction (probability):} &&\hat{y} = \sigma(z) \\
&\text{Objective:} &&\min_{w} \; 
-\sum_{i=1}^{n} \left[
y_i \log \hat{y}_i + (1-y_i)\log(1-\hat{y}_i)
\right]
\end{aligned}
$$

In the previous post, we examined linear regression, where the objective is to predict continuous target values directly from input features. In this post, we turn to a closely related but fundamentally different task: linear classification, where the goal is to assign each input to one of a set of discrete categories.

While both settings rely on the same linear scoring function, classification interprets its output differently. Instead of treating it as a prediction, the linear model produces a real-valued score that is mapped to a class probability through a link function with range (0,1), such as the sigmoid. This change in interpretation leads to a different training objective – maximizing likelihood rather than minimizing squared error – and consequently eliminates the possibility of a closed-form solution.

**link function**

In classification, the output of a linear model cannot be interpreted directly as a class label. Instead, it is treated as a score, which must be mapped to a valid probability range (0, 1) for each label. This is achieved through a link function, which transforms real-valued inputs into probabilities in a manner consistent with the structure of the labels.

*independent binary label(s):* 
When each label represents an independent yes/no decision–such as in binary or multilabel classification–we model each label as a Bernoulli random variable. In this case, for any score $z$ associated with a class, the sigmoid function,

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

is the standard choice and the formulation is called logistic regression. We have, for each input $x$ and the $k_{th}$ label:

$$
p(y_k = 1 \mid x) = \sigma(w_k^\top x) \quad \text{for k = 1, 2, ..., K}
$$

*mutually exclusive labels:* 
When labels are mutually exclusive and each input $x$ belongs to exactly one class, the model must output a valid probability distribution over classes. In this setting, the softmax function is the natural choice of link function, as it maps a vector of real-valued scores to positive probabilities that sum to one. Specifically, given class scores $z \in \mathbb{R}^K$, the softmax is defined as:

$$
\sigma(z)_k = \frac{e^{z_k}}{\sum_{i=1}^{K}e^{z_i}}
$$

and 

$$
p(y = k \mid x)=\frac{e^{\,w_k^\top x}}{\sum_{j=1}^K e^{\,w_j^\top x}}
$$

**objective function**

Unlike linear regression with the mean squared error (MSE) objective, in classification we use the cross-entropy (CE) objective. There are 3 main reasons why this switch is necessary:

1. In linear regression, we assume $y = \hat{y}+\epsilon$ where $\epsilon$ is a Gaussian noise that results naturally from unit peculiarities, measurement errors, etc., for which the MSE is a natural loss function resulting from maximum likelihood estimation. In contrast, classification's assumption is $y \sim Bernoulli(p)$ with $p = \hat{y}$, there is no Gaussian noise, and we only care about the divergence between $\hat{y}$ and the true Bernoulli distribution, for which the cross-entropy loss is the most mathematically appropriate.

2. Cross-entropy penalizes confident but incorrect predictions much more severely than MSE. 
For example, suppose the true label is $y = 0$ while the model predicts $\hat{y} \to 1$. Then,

    $$
    \begin{aligned}
    \text{MSE} 
    &= (y - \hat{y})^2 
    \;\to\; (0 - 1)^2 = 1, \\
    \text{CE} 
    &= -\left[ y \log \hat{y} + (1-y)\log(1-\hat{y}) \right]
    \;\to\; -\log(0) \;\to\; \infty.
    \end{aligned}
    $$

3. With the introduction of a non-linear link function, the MSE loss surface becomes non-convex. This means it can make gradient descent stuck at bad localisations during optimization. Conversely, the logarithms in the cross-entropy loss perfectly "cancel out" the exponential nature of the sigmoid or softmax link functions. This results in a convex loss function, which guarantees a more stable optimization process.

**optimization**

Unlike linear regression, most classification models do not admit a closed-form solution because the model parameters appear inside a nonlinear link function that is applied to a linear combination of the inputs and summed over the dataset. Although the link function is invertible pointwise, the resulting objective function couple the parameters across data points in a way that cannot be solved analytically. As a result, the parameters must be estimated using iterative optimization methods such as gradient descent. We will derive the gradient of logistic regression as an example:

For a single data point $(x, y)$, where $x \in \mathbb{R}^d$ and y $\in$ \{0,1\}, the cross-entropy loss is


$$
\mathcal{L} = - [y \log \sigma(z) + (1-y)\log(1-\sigma(z) ], \qquad z = w^\top x.
$$

The sigmoid function is defined as

$$
\sigma(z) = \frac{1}{1 + e^{-z}},
\qquad \text{and }
\sigma'(z) = \sigma(z)\bigl(1-\sigma(z)\bigr).
$$

Applying the chain rule, we have

$$
\frac{\partial \mathcal{L}}{\partial w} = \frac{\partial \mathcal{L}}{\partial z}\frac{\partial z}{\partial w}.
$$

First, compute the derivative of the loss with respect to $z$:

$$
\begin{aligned}
\frac{\partial \mathcal{L}}{\partial z} &= - \left[y \frac{1}{\sigma(z)} \sigma'(z)+ (1-y)\frac{1}{1-\sigma(z)}(-\sigma'(z))\right]\\
&=- \left[y (1-\sigma(z)) - (1-y)\sigma(z)\right] \\
&=\sigma(z) - y.
\end{aligned}
$$

Since $z = w^\top x$, we have

$$
\frac{\partial z}{\partial w} = x.
$$

Therefore, the gradient of the loss with respect to the parameters for this single data point is

$$
\boxed{
\frac{\partial \mathcal{L}}{\partial w}
=
(\sigma(z) - y)\,x
}
$$

For a dataset of $n$ samples, the full gradient is

$$
\nabla_w \mathcal{L}=X^\top(\hat{y} - y).
$$

Without diving into additional math, for softmax regression paired with cross-entropy loss, the gradient with respect to the logits takes exactly the same form as the above.

Numpy implementations of the above can be found in the [📓 **Colab notebook**](https://colab.research.google.com/drive/1Uu82DXk6Hn4RugOJ5IEjfadOj8hRnUIa?usp=sharing).