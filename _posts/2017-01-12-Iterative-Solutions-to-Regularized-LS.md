---
layout: post
title:  "Iterative Solutions to Regularized LS "
date:   2017-01-12
excerpt: "One solution to regularized LS."
tag:
- statistics 
comments: true
---

As we have proved in the previous post, ridge regression

$$minimize \{ \Vert\boldsymbol{y} - \boldsymbol{X}\boldsymbol{\beta}\Vert_2^2 + \lambda\Vert \boldsymbol{\beta}\Vert_2^2 \}$$

has a solution $$\hat{\boldsymbol{\beta}} = (\boldsymbol{X}^T\boldsymbol{X} + \lambda \boldsymbol{I})^{-1}\boldsymbol{X}^T\boldsymbol{y}$$.

Another regularized least squares, Lasso 

$$minimize \{ \Vert\boldsymbol{y} - \boldsymbol{X}\boldsymbol{\beta}\Vert_2^2 + \lambda\Vert \boldsymbol{\beta}\Vert_1 \}$$

does not have a closed-form solution. Due to the problem that a closed-form algebraic solution may not exist, and more importantly, it could be time consuming to do matrix calculation when $$n$$ is large, iterative algorithms to these optimizations are widely discussed and used. In this post we will discuss the idea of proximal point algorithms and how it solves regularization problems.  

Suppose we have a non-negative regularier $$R(\boldsymbol{\beta})$$ and an algorithm that produces a sequence of iterates and consider the $$k$$th iterate $$\boldsymbol{\beta}_k$$. We can write the objective function as: 

$$
\begin{aligned}
L(\beta) &= \Vert \boldsymbol{y} - \boldsymbol{X}\boldsymbol{\beta}\Vert_2^2 + \lambda R(\boldsymbol{\beta})  \\
&= \Vert \boldsymbol{y} - \boldsymbol{X\beta} _k + \boldsymbol{X\beta} _k - \boldsymbol{X\beta} \Vert _2^2 + \lambda R(\boldsymbol{\beta}) \\
&= \Vert \boldsymbol{y} - \boldsymbol{X\beta} _k \Vert_2^2 + 2(\boldsymbol{y-X\beta}_k)^T\boldsymbol{X} (\boldsymbol{\beta}_k - \boldsymbol{\beta}) + \Vert \boldsymbol{X} (\boldsymbol{\beta}_k - \boldsymbol{\beta}) \Vert_2^2 + \lambda R(\boldsymbol{\beta}) \\
&= C + 2(\boldsymbol{y-X\beta}_k)^T\boldsymbol{X} (\boldsymbol{\beta}_k - \boldsymbol{\beta}) + \Vert \boldsymbol{X} (\boldsymbol{\beta}_k - \boldsymbol{\beta}) \Vert_2^2 + \lambda R(\boldsymbol{\beta}) \\
&\le C + 2(\boldsymbol{y-X\beta}_k)^T\boldsymbol{X} (\boldsymbol{\beta}_k - \boldsymbol{\beta}) + \Vert \boldsymbol{X}\Vert_2^2 \Vert (\boldsymbol{\beta}_k - \boldsymbol{\beta}) \Vert_2^2 + \lambda R(\boldsymbol{\beta}) \\
&\le C + 2(\boldsymbol{y-X\beta}_k)^T\boldsymbol{X} (\boldsymbol{\beta}_k - \boldsymbol{\beta}) + \tau^-1 \Vert (\boldsymbol{\beta}_k - \boldsymbol{\beta}) \Vert_2^2 + \lambda R(\boldsymbol{\beta}) \\
\end{aligned}
$$

where $$0 < \tau < 1/\Vert X \Vert_2^2$$. Observe that the upper bound on the right hand side touches the original objective $$L$$ at the point $$\beta = \beta_k$$. Now we will choose $$\mathbf{\beta}$$ to minimize this bound, that is 

$$
\begin{aligned} 
\boldsymbol{\beta}_{k+1} &= argmin \{ 2(\boldsymbol{y-X\beta}_k)^T\boldsymbol{X} (\boldsymbol{\beta}_k - \boldsymbol{\beta}) + \tau^-1 \Vert (\boldsymbol{\beta}_k - \boldsymbol{\beta}) \Vert_2^2 + \lambda R(\boldsymbol{\beta}) \} \\
&= argmin \{ 2\tau(\boldsymbol{y-X\beta}_k)^T\boldsymbol{X} (\boldsymbol{\beta}_k - \boldsymbol{\beta}) + \Vert (\boldsymbol{\beta}_k - \boldsymbol{\beta}) \Vert_2^2 + \tau\lambda R(\boldsymbol{\beta}) \}
\end{aligned}
$$

Now define $$\boldsymbol{v} = \tau\boldsymbol{X}^T(\boldsymbol{y} - \boldsymbol{X\beta}_k)$$, then we can write the optimization above as 

$$
\begin{aligned}
\boldsymbol{\beta}_{k+1} &= argmin \{ (2\boldsymbol{v}^T\boldsymbol{\beta}_k - \boldsymbol{\beta}) + \Vert \boldsymbol{\beta}_k - \boldsymbol{\beta} \Vert _2^2 + \tau \lambda R( \boldsymbol{\beta}  ) \}  \\
&= argmin \{ \Vert \boldsymbol{v} + \boldsymbol{\beta}_k - \boldsymbol{\beta} \Vert_2^2 - \Vert \boldsymbol{v} \Vert _2^2 + \tau \lambda R(\boldsymbol{\beta})  \\
&= argmin \{ \Vert \boldsymbol{v} + \boldsymbol{\beta}_k - \boldsymbol{\beta} \Vert_2^2 + \tau \lambda R(\boldsymbol{\beta})  \\ 
\end{aligned}
$$

Note that the optimization above is seperable. The solution to the optimization above is called the proximal operator of the function R. So this sort of iterative optimization is often referred to as a proximal point algorithm.  

Consider the ridge regression, $$R(\boldsymbol{\beta}  ) = \Vert \beta \Vert_2^2$$. It could be proved that the update rule of Ridge is:

$$
\boldsymbol{\beta}_{k+1} = \frac{1}{1+\tau\lambda}(\boldsymbol{\beta}_k - \tau \boldsymbol{X}^T(\boldsymbol{X}\boldsymbol{\beta}_k - \boldsymbol{y}))
$$

Here is the implementation of Ridge in MATLAB: 

    function [ b2 ] = Ridge( X, y, tau, diff, lambda, b0 )    
    %(e) Ridge Regression Implementation
    b1 = b0-5; 
    b2 = b0;
    count = 0;
    while (norm(b2-b1) > diff)
        b1 = b2;
        v = tau*X'*(y-X*b1);
        z = b1 + v;
        b2 = z/(1+lambda*tau);
        norm(b2-b1);
    end
    end



For Lasso, which has $$R( \boldsymbol{\beta} ) = \Vert \beta \Vert _1$$, the solution to it is given by the soft-threshold operation:  

$$
\boldsymbol{\beta}_{k+1} = sign(\boldsymbol{v} + \boldsymbol{\beta}_k)(\|\boldsymbol{v} + \boldsymbol{\beta}_k\| - \tau\lambda/2)_{+}
$$

Here is the implementation of Ridge in MATLAB: 
{% highlight matlab %}
    function [ b2 ] = lasso( X, y, tau, diff, lambda, b0)

    %(a) iterative soft-thresholding for Lasso
    b1 = b0-5;
    b2 = b0;
    count = 0;
    while (norm(b2-b1) > diff)
        b1 = b2;
        v = tau*X'*(y-X*b1);
        z = b1 + v;
        for i = 1:length(b2)
            b2(i) = sign(z(i))*max(0, abs(z(i))-tau*lambda/2);
        end
        count = count + 1;
    end
    end
{% endhighlight %}