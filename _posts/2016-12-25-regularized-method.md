---
layout: post
title:  "Regularized Method"
date:   2016-12-25
excerpt: "Something about regularized method."
tag:
- statistics 
comments: true
---

In the problem of linear regression, to prevent overfitting, different kinds of penalty terms are added to the loss function, this is called regularization. Two mostly commonly useded regularized methods are Rigid ($$l_2$$ penalty) and Lasso ($$l_1$$ penalty). 

**Tikhonov regularization (Ridge regression)**. Rigid regression is called $$l_2$$ regularization because a 2-norm of the coefficients are added after the sum of squared error term. Now the loss function becomes:

$$minimize \{ \Vert\boldsymbol{y} - \boldsymbol{X}\boldsymbol{\beta}\Vert_2^2 + \lambda\Vert \boldsymbol{\beta}\Vert_2^2 \}$$

where $$\lambda$$ > 0 is a parameter we choose that determines the relative weight we want to assign to each objective. 

The second term is also called shrinkage penalty, which, as its name suggests, is helpful for shrinking the fitted coefficients. The shrinkage penalty is an important tool of preventing model overfitting. 
Generally, the training error tends to decrease whenever we increases the model complexity. However with too much fitting, the model adapts itself too closey to the training data, and will not generalize well. By adding the $$l_2$$ penalty, ridge regression could help refraining the model fit too much noises of training data, by allowing larger squared bias. 

We could solve the optimization problem by finding an expression for the minimizer $$\hat{\beta}$$. The augmented cost function is an ordinary least-squares problem in disguise. To see why, notice that

$$\Vert\boldsymbol{y}-\boldsymbol{X}\boldsymbol{\beta}\Vert_2^2 + \lambda||\boldsymbol{\beta}||_2^2 = \Vert 
\begin{bmatrix}
\boldsymbol{y}-\boldsymbol{X}\boldsymbol{\beta}   \\
\sqrt{\lambda}\boldsymbol{\beta} 
\end{bmatrix} 
\Vert_2^2 = \Vert 
\begin{bmatrix}
\boldsymbol{y}   \\
0
\end{bmatrix} - 
\begin{bmatrix} 
\boldsymbol{X} \\
\sqrt{\lambda}\boldsymbol{I}
\end{bmatrix} 
\boldsymbol{\beta}\Vert_2^2$$

Applying the least-sqaures formula to these new matrices, we find: 

$$\hat{\boldsymbol{\beta}} = (
\begin{bmatrix} 
\boldsymbol{X} \\
\sqrt{\lambda}\boldsymbol{I}
\end{bmatrix}^T
\begin{bmatrix} 
\boldsymbol{X} \\
\sqrt{\lambda}\boldsymbol{I}
\end{bmatrix} 
)^{-1}
\begin{bmatrix} 
\boldsymbol{X} \\
\sqrt{\lambda}\boldsymbol{I}
\end{bmatrix}^T
\begin{bmatrix} 
\boldsymbol{y} \\
0
\end{bmatrix} = (\boldsymbol{X}^T\boldsymbol{X} + \lambda \boldsymbol{I})^{-1}\boldsymbol{A}^T\boldsymbol{b}
$$

Alternatively, we can find the modified normal equations direcly by differentiating the cost function, and setting the derivative equal to zero. 

**Lasso**. Another commonly used regularization is Lasso. The lasso criterion is:

$$minimize \{ \Vert\boldsymbol{y} - \boldsymbol{X}\boldsymbol{\beta}\Vert_2^2 + \lambda\Vert \boldsymbol{\beta}\Vert_1 \}$$

As we can see the idea here is similar to ridge, except that the penalty term we add to the loss function is one-norm of the coefficients (and thus, Lasso is also called $l_1$ regularization). 

The most interesting property of the Lasso is that the optimiziation above usually has many of the elements of $$\boldsymbol{\beta}$$ set to zero (i.e. features selection). It tends to put weight only on elements of $\boldsymbol{\beta}$ that correspond to the columns of $$\boldsymbol{X}$$ that are most predictive of $\boldsymbol{y}. The larger $$\lambda$$ is, the sparser the solution $$\boldsymbol{\hat{\beta}}$$. Unlike Ridge, we do not have a closed form solution for the Lasso. The solution to the Lasso criterion is given by the "soft-threshold" operation: 

$$\beta_j = sign(y_i)(\|y_i\| - \lambda/2)_{+}$$

Despite of the nonexistence of the closed-form solution to the Lasso, iterative solvers can be used to obtain good approximate to the solution at a lower cost. We choose to use the Landweber iteration to the optimization. It could be proved that the bound optimization algorithm to approximate the Lasso solution is given by 

$$\beta_{k+1} = sign(\boldsymbol{z}_k)(\|\boldsymbol{z}_k\| - \tau\lambda/2)_{+}$$

where $$\boldsymbol{z_k} = \boldsymbol{\beta_k} - \tau \boldsymbol{X^{T}}(\boldsymbol{X\beta_k} - \boldsymbol{y})$$. We will talk more about the iterative solution to regularized LS and how to derive them in future posts.
