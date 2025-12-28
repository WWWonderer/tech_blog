---
title: "ML Basics - Linear Regression Revisited"
date: 2025-12-27
categories: "machine_learning"
---

$$
\begin{aligned}
&\text{Given data:} &&X \in \mathbb{R}^{n \times d}, \; y \in \mathbb{R}^{n \times 1} \\
&\text{Model parameters:} &&w \in \mathbb{R}^{d \times 1} \\
&\text{Prediction:} &&\hat{y} = Xw \\
&\text{Objective:} &&\min_{w} \left\lVert \hat{y} - y \right\rVert_2^2
\end{aligned}
$$

We want to find $w$ such that the predicted $\hat{y}$ is close to the actual $y$. Note that by adding a column of 1 to $X$ and a constant $b$ to $w$, we can incorporate the biases. 

**Closed-Form Derivation**

$$
\begin{aligned}
\left\lVert \hat{y} - y \right\rVert_2^2 &= (Xw - y)^T (Xw - y) \\
&= (w^TX^T - y^T)(Xw - y) \\
&= w^TX^TXw - \underbrace{w^TX^Ty - y^TXw}_{scalars} + y^Ty\\
&= w^TX^TXw - 2w^TX^Ty + y^Ty
\end{aligned}
$$

Since $x^TX^TXw$ is positive semidefinite, it is convex and a closed-form solution exists. To find it, we take derivative to 0:

$$
\begin{aligned}
\nabla_w \left\lVert \hat{y} - y \right\rVert_2^2 &= \nabla_w [w^TX^TXw - 2w^TX^Ty + y^Ty] \\
&= \underbrace{2X^TXw}_{\text{$\nabla_w[w^TAw] = 2Aw$ when $A$ is symmetric}} - 2X^Ty + 0 \\
2X^y &= 2X^TXw \text{ (at 0 derivative)}\\
w &= (X^TX)^{-1}X^Ty
\end{aligned}
$$

**Gradient Descent**

As the closed-form solution requires inverting a matrix, which can be expensive
for large feature dimensions, we can instead approximate $w$ using gradient
descent. Given a learning rate $\eta$, we iteratively update $w$ as:

$$
\begin{aligned}
w_{t+1}
&= w_t - \eta \nabla_w \frac{1}{n} \lVert Xw - y \rVert_2^2 \\
&= w_t - \eta \frac{2}{n} X^T (Xw - y) \\
&= w_t - \eta (X^T X w - X^T y),
\end{aligned}
$$

where constant factors are absorbed into $\eta$.

Numpy implementations of the above can be found in the [📓 **Colab notebook**](https://colab.research.google.com/drive/18JtCzca8PcuG5yQ48zPPQuKCFgodDNij?usp=sharing).
