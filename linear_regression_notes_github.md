# Linear Regression and Multiple Linear Regression

## Comprehensive Notes with Detailed Derivations, Assumptions, Geometry, and the `Ax = b` View

---

# 1. What is Regression?

Regression is a supervised learning technique used when the output to be predicted is a **continuous numerical quantity**.

Examples:

- house price
- temperature
- salary
- demand
- sales
- pressure
- signal strength
- stock value

In regression we observe several input-output pairs and try to learn a mathematical relationship between them.

If one input variable is used, we obtain **simple linear regression**.

If several input variables are used, we obtain **multiple linear regression**.

The central idea of linear regression is:

> Find the line, plane, or hyperplane whose predicted values are as close as possible to the observed values.

The most common fitting criterion is **least squares**.

---

# 2. Simple Linear Regression

Suppose a dataset contains one input variable and one output variable.

For sample `i`, write

$$
y_i = \beta_0 + \beta_1 t_i + \varepsilon_i
$$

where:

- `t_i` = input or predictor for sample `i`
- `y_i` = observed output
- `beta_0` = intercept
- `beta_1` = slope
- `epsilon_i` = random error

I use `t_i` for the scalar input here because later we will use the standard linear-algebra notation

$$
Ax=b
$$

where `x` represents the vector of unknown model coefficients.

The deterministic part of the model is

$$
E[y_i \mid t_i] = \beta_0 + \beta_1 t_i
$$

and the observed value is

$$
y_i = E[y_i \mid t_i] + \varepsilon_i.
$$

---

# 3. Meaning of the Intercept and Slope

The regression line is

$$
\hat y = \hat\beta_0 + \hat\beta_1 t.
$$

## 3.1 Intercept

The intercept is the predicted value of `y` when

$$
t=0.
$$

So

$$
\hat y = \hat\beta_0
$$

when `t = 0`.

The intercept is mathematically necessary in many models even when `t = 0` is not physically meaningful.

---

## 3.2 Slope

The slope measures the change in predicted output for a one-unit increase in the input.

If

$$
\hat\beta_1 = 4,
$$

then increasing `t` by 1 increases the predicted response by 4 units.

If

$$
\hat\beta_1 < 0,
$$

the relationship is decreasing.

---

# 4. Why Do We Need an Error Term?

Real observations almost never lie exactly on one straight line.

Therefore we write

$$
y_i = \beta_0 + \beta_1 t_i + \varepsilon_i.
$$

The term

$$
\varepsilon_i
$$

contains the part of `y_i` that the model does not explain.

It may represent:

- measurement noise
- omitted variables
- natural randomness
- approximation error
- other unmodelled influences

The theoretical error is

$$
\varepsilon_i = y_i - E[y_i \mid t_i].
$$

After fitting the model, we calculate a **residual**

$$
e_i = y_i - \hat y_i.
$$

Error and residual are related, but they are not exactly the same:

- `epsilon_i` depends on the unknown true model
- `e_i` is computed from the fitted model

---

# 5. Ordinary Least Squares

For each observation,

$$
e_i = y_i - \hat y_i.
$$

For simple linear regression,

$$
e_i = y_i - \hat\beta_0 - \hat\beta_1 t_i.
$$

Ordinary Least Squares, or OLS, chooses the coefficients that minimize the **sum of squared residuals**:

$$
S(\beta_0,\beta_1)
=
\sum_{i=1}^{n}
\left(
y_i-\beta_0-\beta_1 t_i
\right)^2.
$$

This quantity is also called:

- SSE = Sum of Squared Errors
- RSS = Residual Sum of Squares

depending on the notation used.

---

# 6. Why Square the Residuals?

Suppose residuals are

$$
2,\ -2.
$$

If we simply add them,

$$
2+(-2)=0.
$$

The errors cancel even though the model is not perfect.

Squaring gives

$$
2^2+(-2)^2=8.
$$

Squaring also penalizes large errors more strongly.

For example,

$$
2^2=4
$$

but

$$
10^2=100.
$$

So least squares strongly discourages very large residuals.

---

# 7. Derivation of Simple Linear Regression

We minimize

$$
S(\beta_0,\beta_1)
=
\sum_{i=1}^{n}
\left(
y_i-\beta_0-\beta_1t_i
\right)^2.
$$

We differentiate with respect to both unknown parameters.

---

## 7.1 Derivative with Respect to the Intercept

Differentiate:

$$
\frac{\partial S}{\partial \beta_0}
=
-2
\sum_{i=1}^{n}
\left(
y_i-\beta_0-\beta_1t_i
\right).
$$

At the minimum,

$$
\frac{\partial S}{\partial \beta_0}=0.
$$

Therefore

$$
\sum_{i=1}^{n}
\left(
y_i-\beta_0-\beta_1t_i
\right)=0.
$$

Expand:

$$
\sum y_i
-
n\beta_0
-
\beta_1\sum t_i
=
0.
$$

So

$$
n\beta_0
=
\sum y_i
-
\beta_1\sum t_i.
$$

Divide by `n`:

$$
\beta_0
=
\bar y-\beta_1\bar t.
$$

Therefore the fitted intercept is

$$
\hat\beta_0
=
\bar y-\hat\beta_1\bar t.
$$

This immediately implies that the fitted line passes through

$$
(\bar t,\bar y).
$$

---

## 7.2 Derivative with Respect to the Slope

Differentiate:

$$
\frac{\partial S}{\partial \beta_1}
=
-2
\sum_{i=1}^{n}
t_i
\left(
y_i-\beta_0-\beta_1t_i
\right).
$$

Set this equal to zero:

$$
\sum_{i=1}^{n}
t_i
\left(
y_i-\beta_0-\beta_1t_i
\right)=0.
$$

Substitute

$$
\beta_0=\bar y-\beta_1\bar t.
$$

Then

$$
y_i-\beta_0-\beta_1t_i
=
y_i-\bar y-\beta_1(t_i-\bar t).
$$

Therefore

$$
\sum t_i
\left[
(y_i-\bar y)
-
\beta_1(t_i-\bar t)
\right]
=
0.
$$

After centering the expression, we obtain

$$
\sum
(t_i-\bar t)(y_i-\bar y)
=
\beta_1
\sum
(t_i-\bar t)^2.
$$

Hence

$$
\hat\beta_1
=
\frac{
\sum_{i=1}^{n}
(t_i-\bar t)(y_i-\bar y)
}{
\sum_{i=1}^{n}
(t_i-\bar t)^2
}.
$$

Then

$$
\hat\beta_0
=
\bar y-\hat\beta_1\bar t.
$$

These are the ordinary least-squares estimators for simple linear regression.

---

# 8. Interpretation of the Simple Regression Slope

Define

$$
S_{tt}
=
\sum_{i=1}^{n}(t_i-\bar t)^2
$$

and

$$
S_{ty}
=
\sum_{i=1}^{n}(t_i-\bar t)(y_i-\bar y).
$$

Then

$$
\hat\beta_1
=
\frac{S_{ty}}{S_{tt}}.
$$

Using sample covariance and sample variance,

$$
\hat\beta_1
=
\frac{\operatorname{Cov}(t,y)}
{\operatorname{Var}(t)}.
$$

Also, if `r_ty` is the sample correlation,

$$
\hat\beta_1
=
r_{ty}\frac{s_y}{s_t}.
$$

So:

- covariance determines the direction of the relationship
- predictor variance scales the slope
- correlation gives the direction and strength after removing units

---

# 9. From Simple Regression to Multiple Linear Regression

Suppose we have several predictors.

For observation `i`,

$$
y_i
=
\beta_0
+
\beta_1 z_{i1}
+
\beta_2 z_{i2}
+
\cdots
+
\beta_k z_{ik}
+
\varepsilon_i.
$$

Examples:

For house price,

- `z_1` = area
- `z_2` = number of bedrooms
- `z_3` = age
- `z_4` = distance from city center

The model is still called **linear** because it is linear in the coefficients:

$$
\beta_0,\beta_1,\ldots,\beta_k.
$$

The predictors themselves can be transformed.

For example,

$$
y
=
\beta_0+\beta_1z+\beta_2z^2+\varepsilon
$$

is still linear regression because the unknown coefficients appear linearly.

---

# 10. Multiple Regression in `Ax = b` Form

This is the most useful linear-algebra representation.

Suppose there are:

- `n` observations
- `p` unknown coefficients, including the intercept

Write

$$
Ax=b+\text{error}
$$

or more precisely

$$
b=Ax+\varepsilon.
$$

Here:

$$
A \in \mathbb R^{n\times p}
$$

is the design matrix,

$$
x \in \mathbb R^{p}
$$

is the unknown coefficient vector,

and

$$
b \in \mathbb R^{n}
$$

is the observed output vector.

A typical design matrix is

$$
A=
\begin{bmatrix}
1 & z_{11} & z_{12} & \cdots & z_{1,p-1}\\
1 & z_{21} & z_{22} & \cdots & z_{2,p-1}\\
\vdots & \vdots & \vdots & \ddots & \vdots\\
1 & z_{n1} & z_{n2} & \cdots & z_{n,p-1}
\end{bmatrix}.
$$

The coefficient vector is

$$
x=
\begin{bmatrix}
\beta_0\\
\beta_1\\
\vdots\\
\beta_{p-1}
\end{bmatrix}.
$$

The output vector is

$$
b=
\begin{bmatrix}
y_1\\
y_2\\
\vdots\\
y_n
\end{bmatrix}.
$$

Therefore

$$
Ax
=
\begin{bmatrix}
\hat y_1\\
\hat y_2\\
\vdots\\
\hat y_n
\end{bmatrix}.
$$

So in regression:

$$
\hat b=Ax.
$$

---

# 11. What Does Each Row of `A` Mean?

Row `i` contains all predictors for observation `i`.

For example,

$$
a_i^T
=
\begin{bmatrix}
1 & z_{i1} & z_{i2}
\end{bmatrix}.
$$

Then

$$
\hat y_i
=
a_i^T x.
$$

So each row produces one prediction.

---

# 12. What Does Each Column of `A` Mean?

Write

$$
A=
\begin{bmatrix}
| & | & & |\\
a_1 & a_2 & \cdots & a_p\\
| & | & & |
\end{bmatrix}.
$$

Each column is a vector in

$$
\mathbb R^n.
$$

For example, a feature column is

$$
a_j
=
\begin{bmatrix}
z_{1j}\\
z_{2j}\\
\vdots\\
z_{nj}
\end{bmatrix}.
$$

It contains the values of one feature across all observations.

Now multiply:

$$
Ax
=
x_1a_1+x_2a_2+\cdots+x_pa_p.
$$

Therefore every possible prediction vector is a linear combination of the columns of `A`.

Hence

$$
Ax \in \operatorname{Col}(A).
$$

---

# 13. Column Space

The column space is

$$
\operatorname{Col}(A)
=
\operatorname{span}
\{a_1,a_2,\ldots,a_p\}.
$$

This is the set of all output vectors the model can produce.

The target vector `b` can be any vector in

$$
\mathbb R^n,
$$

but the model can only produce vectors inside

$$
\operatorname{Col}(A).
$$

This is the central geometric idea of least squares.

---

# 14. Why Regression Is Usually an Approximate `Ax = b` Problem

If we could solve

$$
Ax=b
$$

exactly, then every observation would be fitted perfectly.

But in most datasets,

$$
b\notin\operatorname{Col}(A).
$$

Therefore there is no exact coefficient vector `x` satisfying

$$
Ax=b.
$$

Instead, we find

$$
\hat x
$$

such that

$$
A\hat x
$$

is as close as possible to `b`.

Thus least squares solves

$$
\hat x
=
\arg\min_x
\|b-Ax\|_2^2.
$$

---

# 15. Residual Vector in `Ax = b` Form

Define

$$
e=b-A\hat x.
$$

Then

$$
b=A\hat x+e.
$$

Interpretation:

- `A hat{x}` = explained or fitted component
- `e` = unexplained residual component

OLS minimizes

$$
\|e\|_2^2.
$$

Since

$$
\|e\|_2^2
=
e^Te,
$$

we minimize

$$
(b-Ax)^T(b-Ax).
$$

---

# 16. Least Squares as Geometry

The fitted vector

$$
A\hat x
$$

must lie in

$$
\operatorname{Col}(A).
$$

The target vector `b` is generally outside that subspace.

The nearest point in a subspace is obtained by an **orthogonal projection**.

Therefore the residual

$$
e=b-A\hat x
$$

must be perpendicular to the column space of `A`.

So

$$
e\perp\operatorname{Col}(A).
$$

Since every column `a_j` lies in the column space,

$$
a_j^Te=0
$$

for every `j`.

Putting all these equations together gives

$$
A^Te=0.
$$

Substitute

$$
e=b-A\hat x.
$$

Then

$$
A^T(b-A\hat x)=0.
$$

Expand:

$$
A^Tb-A^TA\hat x=0.
$$

Therefore

$$
A^TA\hat x=A^Tb.
$$

These are the **normal equations**.

If `A^T A` is invertible,

$$
\hat x
=
(A^TA)^{-1}A^Tb.
$$

---

# 17. Why the Residual Must Be Orthogonal: Detailed Geometric Proof

Let

$$
p=A\hat x
$$

be the OLS prediction.

Take any other possible prediction

$$
q=Ax.
$$

Both `p` and `q` belong to

$$
\operatorname{Col}(A).
$$

Therefore

$$
q-p
$$

also lies in the column space.

The OLS residual is

$$
e=b-p.
$$

Suppose

$$
e\perp\operatorname{Col}(A).
$$

Then

$$
e\perp(q-p).
$$

Now

$$
b-q
=
(b-p)+(p-q).
$$

These two terms are perpendicular.

Therefore, by the Pythagorean theorem,

$$
\|b-q\|^2
=
\|b-p\|^2
+
\|p-q\|^2.
$$

Since

$$
\|p-q\|^2\ge0,
$$

we have

$$
\|b-q\|^2
\ge
\|b-p\|^2.
$$

Therefore `p` is the closest point in the entire column space.

Hence the OLS prediction is exactly the orthogonal projection of `b` onto

$$
\operatorname{Col}(A).
$$

---

# 18. Visualization with Three Observations

Suppose

$$
n=3.
$$

Then every column of `A` is a vector in

$$
\mathbb R^3.
$$

If there is one independent feature column, the column space is a line.

If there are two linearly independent feature columns, the column space is a plane.

The target

$$
b\in\mathbb R^3
$$

is a point or vector somewhere in the 3D room.

OLS projects `b` onto the line or plane.

The projection is

$$
A\hat x.
$$

The perpendicular gap is

$$
e=b-A\hat x.
$$

So:

$$
b
=
A\hat x+e.
$$

This is an orthogonal decomposition.

---

# 19. Why the Space Has One Coordinate per Observation

Suppose

$$
b=
\begin{bmatrix}
y_1\\
y_2\\
y_3
\end{bmatrix}.
$$

This vector needs three coordinates, so it belongs to

$$
\mathbb R^3.
$$

For `n` observations,

$$
b=
\begin{bmatrix}
y_1\\
y_2\\
\vdots\\
y_n
\end{bmatrix},
$$

so

$$
b\in\mathbb R^n.
$$

Every feature column also has `n` entries, so each feature column is also a vector in

$$
\mathbb R^n.
$$

That is why regression geometry takes place in an `n`-dimensional observation space.

---

# 20. Ordinary Data Plot vs Column-Space Geometry

These two pictures must not be confused.

## 20.1 Ordinary scatterplot

For one feature, we draw points

$$
(t_i,y_i)
$$

in a 2D plane.

The regression line is

$$
\hat y=\hat\beta_0+\hat\beta_1t.
$$

The ordinary OLS residual is

$$
e_i=y_i-\hat y_i.
$$

This is a **vertical distance**.

OLS does not generally minimize perpendicular distance from each data point to the plotted regression line.

---

## 20.2 Column-space picture

Now we collect all target values into one vector:

$$
b=
\begin{bmatrix}
y_1\\
y_2\\
\vdots\\
y_n
\end{bmatrix}.
$$

The prediction is

$$
A\hat x.
$$

Here the residual vector

$$
e=b-A\hat x
$$

is genuinely perpendicular to the complete model subspace:

$$
e\perp\operatorname{Col}(A).
$$

Therefore:

- residuals are vertical in the ordinary predictor-response plot
- the full residual vector is orthogonal in the observation-space picture

There is no contradiction because these are different geometric spaces.

---

# 21. Matrix-Calculus Derivation of Least Squares

Define

$$
J(x)
=
\|b-Ax\|_2^2.
$$

Write the squared norm as a dot product:

$$
J(x)
=
(b-Ax)^T(b-Ax).
$$

Expand:

$$
J(x)
=
b^Tb
-
b^TAx
-
x^TA^Tb
+
x^TA^TAx.
$$

The middle two terms are equal because they are scalars:

$$
b^TAx
=
x^TA^Tb.
$$

Therefore

$$
J(x)
=
b^Tb
-
2x^TA^Tb
+
x^TA^TAx.
$$

Differentiate with respect to `x`:

$$
\nabla_xJ
=
-2A^Tb
+
2A^TAx.
$$

At the minimum,

$$
\nabla_xJ=0.
$$

Therefore

$$
-2A^Tb+2A^TA\hat x=0.
$$

Divide by 2:

$$
A^TA\hat x=A^Tb.
$$

Thus

$$
\hat x
=
(A^TA)^{-1}A^Tb
$$

when the inverse exists.

This is exactly the same result obtained from geometry.

---

# 22. Why `A^T A` Appears

The original system is

$$
Ax=b.
$$

When `b` is outside the column space, no exact solution exists.

Multiplying by `A^T` gives

$$
A^TAx=A^Tb.
$$

Geometrically this is not an arbitrary trick.

It enforces

$$
A^T(b-Ax)=0,
$$

meaning that the residual is orthogonal to every column of `A`.

So the normal equations are simply the algebraic form of orthogonal projection.

---

# 23. Dimensions of All Important Matrices

Suppose

$$
A\in\mathbb R^{n\times p}.
$$

Then

$$
x\in\mathbb R^{p\times1},
$$

$$
b\in\mathbb R^{n\times1}.
$$

Also,

$$
A^T\in\mathbb R^{p\times n}.
$$

Therefore

$$
A^TA
$$

has size

$$
(p\times n)(n\times p)
=
p\times p.
$$

And

$$
A^Tb
$$

has size

$$
(p\times n)(n\times1)
=
p\times1.
$$

Hence

$$
(A^TA)^{-1}A^Tb
$$

has size

$$
(p\times p)(p\times1)
=
p\times1.
$$

That matches the required size of the coefficient vector.

---

# 24. When Is `A^T A` Invertible?

The matrix

$$
A^TA
$$

is invertible when the columns of `A` are linearly independent.

Equivalently,

$$
\operatorname{rank}(A)=p.
$$

To see why, let `v` be any vector.

Then

$$
v^TA^TAv
=
(Av)^T(Av).
$$

Therefore

$$
v^TA^TAv
=
\|Av\|^2.
$$

If the columns of `A` are linearly independent, then

$$
Av=0
$$

only when

$$
v=0.
$$

So for every nonzero `v`,

$$
\|Av\|^2>0.
$$

Thus

$$
v^TA^TAv>0.
$$

Therefore `A^T A` is positive definite and hence invertible.

---

# 25. Rank and Dimension

The column-space dimension is

$$
\dim(\operatorname{Col}(A))
=
\operatorname{rank}(A).
$$

Also,

$$
\operatorname{rank}(A)
\le
\min(n,p).
$$

If all `p` columns are independent and `n > p`, then

$$
\operatorname{rank}(A)=p.
$$

Therefore the model can move only inside a `p`-dimensional subspace of the full `n`-dimensional output space.

---

# 26. Overdetermined System

If

$$
n>p,
$$

there are more equations than unknown coefficients.

For example:

$$
n=1000,
\qquad
p=5.
$$

Then

$$
A\in\mathbb R^{1000\times5}.
$$

If the columns are independent,

$$
\operatorname{Col}(A)
$$

has dimension 5 inside

$$
\mathbb R^{1000}.
$$

A general target vector `b` will not lie exactly in this small subspace.

Therefore least squares finds the projection.

This is the most common classical regression situation.

---

# 27. Underdetermined System

If

$$
n<p,
$$

there are more coefficients than observations.

Then the coefficient vector may not be uniquely determined.

If the rows of `A` have full rank `n`, then the model can represent every vector in

$$
\mathbb R^n.
$$

In that case there may be infinitely many solutions to

$$
Ax=b.
$$

To choose among them, one may use:

- minimum-norm solution
- pseudoinverse
- ridge regression
- other regularization methods

---

# 28. Moore-Penrose Pseudoinverse

If the ordinary inverse formula cannot be used, least squares can be written generally as

$$
\hat x=A^+b
$$

where

$$
A^+
$$

is the Moore-Penrose pseudoinverse.

When `A` has full column rank,

$$
A^+
=
(A^TA)^{-1}A^T.
$$

So the ordinary formula is a special case.

---

# 29. Projection Matrix

Start from

$$
\hat x
=
(A^TA)^{-1}A^Tb.
$$

Then

$$
\hat b
=
A\hat x.
$$

Substitute:

$$
\hat b
=
A(A^TA)^{-1}A^Tb.
$$

Define

$$
P
=
A(A^TA)^{-1}A^T.
$$

Then

$$
\hat b=Pb.
$$

`P` is the orthogonal projection matrix onto

$$
\operatorname{Col}(A).
$$

In regression this is also called the **hat matrix** because

$$
\hat b=Pb.
$$

---

# 30. Symmetry of the Projection Matrix

We have

$$
P
=
A(A^TA)^{-1}A^T.
$$

Take the transpose:

$$
P^T
=
\left[
A(A^TA)^{-1}A^T
\right]^T.
$$

Reverse the order:

$$
P^T
=
A
\left[
(A^TA)^{-1}
\right]^T
A^T.
$$

Since

$$
A^TA
$$

is symmetric, its inverse is symmetric.

Therefore

$$
P^T=P.
$$

So the orthogonal projection matrix is symmetric.

---

# 31. Idempotence of the Projection Matrix

Compute

$$
P^2
=
A(A^TA)^{-1}A^T
A(A^TA)^{-1}A^T.
$$

Group the middle terms:

$$
P^2
=
A(A^TA)^{-1}
(A^TA)
(A^TA)^{-1}
A^T.
$$

Therefore

$$
P^2
=
A(A^TA)^{-1}A^T.
$$

Hence

$$
P^2=P.
$$

Interpretation:

> Once a vector has been projected onto the model subspace, projecting it again does nothing.

---

# 32. Residual-Maker Matrix

Since

$$
e=b-\hat b
$$

and

$$
\hat b=Pb,
$$

we obtain

$$
e=(I-P)b.
$$

Define

$$
M=I-P.
$$

Then

$$
e=Mb.
$$

`M` projects onto the space perpendicular to

$$
\operatorname{Col}(A).
$$

Important identities:

$$
M^T=M,
$$

$$
M^2=M,
$$

$$
PA=A,
$$

$$
MA=0.
$$

---

# 33. Fundamental Orthogonality Results

OLS gives

$$
A^Te=0.
$$

This single equation produces several important results.

---

## 33.1 Residual Is Orthogonal to Every Predictor Column

If `a_j` is column `j` of `A`,

$$
a_j^Te=0.
$$

So each included feature column is orthogonal to the residual vector.

---

## 33.2 Residual Is Orthogonal to the Fitted Vector

The fitted vector is

$$
\hat b=A\hat x.
$$

Therefore

$$
\hat b^Te
=
\hat x^TA^Te.
$$

But

$$
A^Te=0.
$$

Hence

$$
\hat b^Te=0.
$$

So

$$
\hat b\perp e.
$$

---

## 33.3 Residuals Sum to Zero When an Intercept Is Included

If the first column of `A` is

$$
\mathbf 1
=
\begin{bmatrix}
1\\
1\\
\vdots\\
1
\end{bmatrix},
$$

then

$$
A^Te=0
$$

contains the equation

$$
\mathbf 1^Te=0.
$$

Therefore

$$
\sum_{i=1}^{n}e_i=0.
$$

Thus

$$
\bar e=0.
$$

---

## 33.4 Mean Fitted Output Equals Mean Observed Output

Since

$$
e_i=y_i-\hat y_i
$$

and

$$
\sum e_i=0,
$$

we have

$$
\sum y_i
=
\sum \hat y_i.
$$

Therefore

$$
\bar y
=
\overline{\hat y}.
$$

This property assumes an intercept is included.

---

# 34. Orthogonal Decomposition

Every target vector can be decomposed as

$$
b=\hat b+e.
$$

Here

$$
\hat b\in\operatorname{Col}(A)
$$

and

$$
e\in\operatorname{Null}(A^T).
$$

Also,

$$
\operatorname{Null}(A^T)
=
\operatorname{Col}(A)^\perp.
$$

Therefore

$$
\mathbb R^n
=
\operatorname{Col}(A)
\oplus
\operatorname{Null}(A^T).
$$

The symbol `oplus` means an orthogonal direct sum.

So the output space splits into:

1. a part the regression model can represent
2. a perpendicular part it cannot represent

---

# 35. Proof that `Null(A^T) = Col(A)^perp`

Suppose

$$
z\in\operatorname{Null}(A^T).
$$

Then

$$
A^Tz=0.
$$

This means

$$
a_j^Tz=0
$$

for every column `a_j`.

So `z` is perpendicular to every column of `A`.

Any vector in the column space can be written as

$$
v=c_1a_1+c_2a_2+\cdots+c_pa_p.
$$

Then

$$
z^Tv
=
c_1z^Ta_1
+
c_2z^Ta_2
+
\cdots
+
c_pz^Ta_p.
$$

Every term is zero.

Hence

$$
z^Tv=0.
$$

Therefore

$$
z\perp\operatorname{Col}(A).
$$

Thus

$$
\operatorname{Null}(A^T)
=
\operatorname{Col}(A)^\perp.
$$

---

# 36. Why the Equations Are Called Normal Equations

The normal equations are

$$
A^TA\hat x=A^Tb.
$$

They are equivalent to

$$
A^T(b-A\hat x)=0.
$$

So

$$
A^Te=0.
$$

The residual is **normal**, meaning perpendicular, to the model subspace.

That is why they are called the normal equations.

---

# 37. Linear Regression Assumptions

The assumptions should be separated carefully because different assumptions are needed for different conclusions.

Some assumptions are needed for the model to make sense.

Some are needed for unbiasedness.

Some are needed for the classical variance formula.

Some are needed for exact hypothesis tests.

---

# 38. Assumption 1: Linearity in Parameters

The model has the form

$$
b=Ax+\varepsilon.
$$

Equivalently,

$$
E[b\mid A]=Ax.
$$

The model must be linear in the unknown coefficient vector `x`.

This does **not** mean the relationship must always be a straight line in the original raw variables.

For example,

$$
y
=
\beta_0
+
\beta_1z
+
\beta_2z^2
+
\varepsilon
$$

is linear in the parameters.

Similarly,

$$
y
=
\beta_0
+
\beta_1\log z
+
\varepsilon
$$

is also linear in the parameters.

---

# 39. Assumption 2: No Perfect Multicollinearity

The predictor columns should not be exact linear combinations of one another.

For example, if

$$
a_3=2a_1+5a_2,
$$

then one column is redundant.

In that case,

$$
\operatorname{rank}(A)<p
$$

and

$$
A^TA
$$

is singular.

For the unique classical OLS coefficient formula, we require

$$
\operatorname{rank}(A)=p.
$$

---

# 40. Assumption 3: Zero Conditional Mean

The central exogeneity assumption is

$$
E[\varepsilon\mid A]=0.
$$

This means that after the included predictors are known, the remaining error has no systematic positive or negative component.

Intuitively:

> The regressors should not be systematically related to the omitted part of the outcome.

This assumption is crucial for unbiased estimation.

---

# 41. Why Zero Conditional Mean Matters

From the model,

$$
b=Ax+\varepsilon.
$$

The OLS estimator is

$$
\hat x
=
(A^TA)^{-1}A^Tb.
$$

Substitute the model:

$$
\hat x
=
(A^TA)^{-1}A^T(Ax+\varepsilon).
$$

Expand:

$$
\hat x
=
(A^TA)^{-1}A^TAx
+
(A^TA)^{-1}A^T\varepsilon.
$$

Thus

$$
\hat x
=
x
+
(A^TA)^{-1}A^T\varepsilon.
$$

Take conditional expectation:

$$
E[\hat x\mid A]
=
x
+
(A^TA)^{-1}A^T
E[\varepsilon\mid A].
$$

If

$$
E[\varepsilon\mid A]=0,
$$

then

$$
E[\hat x\mid A]=x.
$$

Therefore OLS is unbiased.

---

# 42. Assumption 4: Homoskedasticity

Homoskedasticity means that every error has the same conditional variance:

$$
\operatorname{Var}(\varepsilon_i\mid A)
=
\sigma^2.
$$

In matrix form, when errors are also uncorrelated,

$$
\operatorname{Var}(\varepsilon\mid A)
=
\sigma^2I.
$$

The opposite is heteroskedasticity.

With heteroskedasticity,

$$
\operatorname{Var}(\varepsilon_i\mid A)
$$

changes from observation to observation.

Homoskedasticity is not required to calculate OLS coefficients.

It is used in the classical OLS variance formula and the standard Gauss-Markov efficiency result.

---

# 43. Assumption 5: No Error Correlation

For different observations,

$$
\operatorname{Cov}(\varepsilon_i,\varepsilon_j\mid A)=0
$$

for

$$
i\ne j.
$$

This assumption can fail in:

- time-series data
- repeated measurements
- spatial data
- clustered observations

When errors are correlated, ordinary OLS standard errors are generally not correct even if the coefficient estimator remains unbiased under suitable exogeneity conditions.

---

# 44. Assumption 6: Normality

For exact classical finite-sample inference, we often assume

$$
\varepsilon\mid A
\sim
N(0,\sigma^2I).
$$

Then

$$
b\mid A
\sim
N(Ax,\sigma^2I).
$$

Normality is **not** needed to derive least squares itself.

It is mainly useful for:

- exact `t` tests
- exact `F` tests
- exact confidence intervals
- maximum-likelihood interpretation

---

# 45. Summary of Which Assumption Is Needed for What

## To compute OLS

We minimize

$$
\|b-Ax\|^2.
$$

No probability distribution is required.

## For a unique inverse formula

Need

$$
\operatorname{rank}(A)=p.
$$

## For unbiasedness

Need

$$
E[\varepsilon\mid A]=0.
$$

## For the classical variance formula

Need

$$
\operatorname{Var}(\varepsilon\mid A)=\sigma^2I.
$$

## For Gauss-Markov BLUE result

Need the linear model, exogeneity, full column rank, and spherical error covariance.

## For exact finite-sample `t` and `F` results

Normality is additionally assumed.

---

# 46. Unbiasedness of OLS

We already obtained

$$
\hat x
=
x
+
(A^TA)^{-1}A^T\varepsilon.
$$

Taking conditional expectation:

$$
E[\hat x\mid A]
=
x+
(A^TA)^{-1}A^T
E[\varepsilon\mid A].
$$

Under

$$
E[\varepsilon\mid A]=0,
$$

we obtain

$$
E[\hat x\mid A]=x.
$$

Therefore

$$
\hat x
$$

is an unbiased estimator of the true coefficient vector.

---

# 47. Variance of the OLS Estimator

Start from

$$
\hat x-x
=
(A^TA)^{-1}A^T\varepsilon.
$$

Define

$$
C=(A^TA)^{-1}A^T.
$$

Then

$$
\hat x-x=C\varepsilon.
$$

Therefore

$$
\operatorname{Var}(\hat x\mid A)
=
C
\operatorname{Var}(\varepsilon\mid A)
C^T.
$$

Under homoskedastic uncorrelated errors,

$$
\operatorname{Var}(\varepsilon\mid A)
=
\sigma^2I.
$$

Hence

$$
\operatorname{Var}(\hat x\mid A)
=
\sigma^2CC^T.
$$

Now

$$
CC^T
=
(A^TA)^{-1}
A^TA
(A^TA)^{-1}.
$$

Therefore

$$
CC^T
=
(A^TA)^{-1}.
$$

Thus

$$
\operatorname{Var}(\hat x\mid A)
=
\sigma^2(A^TA)^{-1}.
$$

This is one of the most important formulas in linear regression.

---

# 48. Meaning of the OLS Variance Formula

The covariance matrix is

$$
\sigma^2(A^TA)^{-1}.
$$

This tells us that uncertainty in the coefficient estimates depends on:

1. noise level `sigma^2`
2. number of observations
3. spread of the predictors
4. correlation among predictors

If predictor columns are nearly linearly dependent, then `A^T A` becomes poorly conditioned and coefficient variances become large.

---

# 49. Gauss-Markov Theorem

The Gauss-Markov theorem states:

> Under the classical linear-model assumptions, OLS is the Best Linear Unbiased Estimator.

This is abbreviated as

$$
BLUE.
$$

It means:

- **Linear**: estimator is linear in `b`
- **Unbiased**: expected value equals the true coefficient vector
- **Best**: it has minimum covariance among linear unbiased estimators

---

# 50. Proof of the Gauss-Markov Theorem

The OLS estimator can be written as

$$
\hat x=Cb
$$

where

$$
C=(A^TA)^{-1}A^T.
$$

Now consider another linear unbiased estimator

$$
\tilde x=Db.
$$

For unbiasedness,

$$
E[\tilde x\mid A]
=
DAx
=
x.
$$

Therefore

$$
DA=I.
$$

Also OLS satisfies

$$
CA=I.
$$

Write

$$
D=C+F.
$$

Then

$$
DA=(C+F)A.
$$

Because both `DA` and `CA` equal `I`,

$$
FA=0.
$$

Now

$$
\operatorname{Var}(\tilde x\mid A)
=
\sigma^2DD^T.
$$

Substitute

$$
D=C+F.
$$

Then

$$
DD^T
=
CC^T
+
CF^T
+
FC^T
+
FF^T.
$$

We have

$$
CF^T
=
(A^TA)^{-1}A^TF^T.
$$

But

$$
A^TF^T=(FA)^T=0.
$$

Therefore

$$
CF^T=0.
$$

Similarly,

$$
FC^T=0.
$$

Thus

$$
\operatorname{Var}(\tilde x\mid A)
=
\operatorname{Var}(\hat x\mid A)
+
\sigma^2FF^T.
$$

Since

$$
FF^T
$$

is positive semidefinite,

$$
\operatorname{Var}(\tilde x\mid A)
-
\operatorname{Var}(\hat x\mid A)
$$

is positive semidefinite.

Therefore no other linear unbiased estimator has a smaller covariance matrix than OLS.

Hence OLS is BLUE.

---

# 51. Estimating the Error Variance

The residual sum of squares is

$$
RSS=e^Te.
$$

If there are `n` observations and `p` estimated coefficients, the unbiased estimator of the error variance is

$$
s^2
=
\frac{RSS}{n-p}.
$$

Why `n-p`?

Because fitting `p` independent coefficients uses `p` degrees of freedom.

The residual space has dimension

$$
n-p
$$

when `A` has full column rank.

---

# 52. Why the Residual Space Has Dimension `n-p`

The column space has dimension

$$
p.
$$

The full observation space has dimension

$$
n.
$$

The orthogonal residual space is

$$
\operatorname{Null}(A^T).
$$

By rank-nullity,

$$
\dim(\operatorname{Null}(A^T))
=
n-p.
$$

That is why residual degrees of freedom are

$$
n-p.
$$

---

# 53. Sum of Squares Decomposition

When an intercept is included, define:

Total Sum of Squares:

$$
TSS
=
\sum_{i=1}^{n}
(y_i-\bar y)^2.
$$

Residual Sum of Squares:

$$
RSS
=
\sum_{i=1}^{n}
(y_i-\hat y_i)^2.
$$

Explained Sum of Squares:

$$
ESS
=
\sum_{i=1}^{n}
(\hat y_i-\bar y)^2.
$$

Then

$$
TSS=ESS+RSS.
$$

---

# 54. Proof of `TSS = ESS + RSS`

Write

$$
y_i-\bar y
=
(\hat y_i-\bar y)
+
(y_i-\hat y_i).
$$

In vector form,

$$
b-\bar y\mathbf 1
=
(\hat b-\bar y\mathbf 1)+e.
$$

The fitted centered component belongs to

$$
\operatorname{Col}(A),
$$

and the residual is perpendicular to that space.

Therefore the two components are orthogonal.

By the Pythagorean theorem,

$$
\|b-\bar y\mathbf 1\|^2
=
\|\hat b-\bar y\mathbf 1\|^2
+
\|e\|^2.
$$

Hence

$$
TSS=ESS+RSS.
$$

---

# 55. Coefficient of Determination `R^2`

Define

$$
R^2
=
1-\frac{RSS}{TSS}.
$$

Using

$$
TSS=ESS+RSS,
$$

we also have

$$
R^2
=
\frac{ESS}{TSS}.
$$

Interpretation:

> `R^2` is the fraction of the total variation in the observed response explained by the fitted regression model.

For a standard regression with an intercept,

$$
0\le R^2\le1.
$$

---

# 56. Adjusted `R^2`

Adding more predictors cannot increase RSS.

Therefore ordinary `R^2` can increase even when a new predictor is not useful.

Adjusted `R^2` introduces a penalty for model size:

$$
\bar R^2
=
1
-
\frac{RSS/(n-p)}
{TSS/(n-1)}.
$$

Adjusted `R^2` may decrease when an unnecessary variable is added.

---

# 57. Standard Error of the Coefficients

The estimated covariance matrix is

$$
\widehat{\operatorname{Var}}(\hat x)
=
s^2(A^TA)^{-1}.
$$

If coefficient `j` corresponds to diagonal element `j,j`, then

$$
SE(\hat x_j)
=
s
\sqrt{
[(A^TA)^{-1}]_{jj}
}.
$$

A larger standard error means the coefficient is estimated less precisely.

---

# 58. Hypothesis Testing for a Coefficient

To test

$$
H_0:x_j=c,
$$

use

$$
t
=
\frac{
\hat x_j-c
}{
SE(\hat x_j)
}.
$$

The common test

$$
H_0:x_j=0
$$

asks whether the corresponding predictor contributes statistically detectable information after controlling for the other predictors.

Under the classical normal model,

$$
t
$$

follows a Student `t` distribution with

$$
n-p
$$

degrees of freedom.

---

# 59. Confidence Interval for a Coefficient

A two-sided confidence interval has the form

$$
\hat x_j
\pm
t_{\alpha/2,n-p}
SE(\hat x_j).
$$

The critical `t` value depends on:

- confidence level
- residual degrees of freedom

---

# 60. Joint Significance and the F Test

Sometimes we test several coefficients together.

For example,

$$
H_0:
x_2=x_3=x_4=0.
$$

An `F` test compares:

- a restricted model
- an unrestricted model

One common form is

$$
F
=
\frac{
(RSS_R-RSS_U)/q
}{
RSS_U/(n-p)
},
$$

where:

- `RSS_R` = restricted-model residual sum of squares
- `RSS_U` = unrestricted-model residual sum of squares
- `q` = number of restrictions

---

# 61. Gaussian Maximum-Likelihood View

Suppose

$$
\varepsilon
\sim
N(0,\sigma^2I).
$$

Then

$$
b\mid A
\sim
N(Ax,\sigma^2I).
$$

The likelihood is proportional to

$$
\exp
\left[
-\frac{1}{2\sigma^2}
(b-Ax)^T(b-Ax)
\right].
$$

Taking the log gives

$$
\log L
=
\text{constant}
-
\frac{1}{2\sigma^2}
\|b-Ax\|^2.
$$

For fixed `sigma^2`, maximizing the likelihood is equivalent to minimizing

$$
\|b-Ax\|^2.
$$

Therefore under Gaussian noise,

$$
\hat x_{MLE}
=
\hat x_{OLS}.
$$

This explains why squared-error loss arises naturally from a Gaussian error model.

---

# 62. Prediction for a New Observation

Suppose a new observation has feature vector

$$
a_0.
$$

The predicted mean response is

$$
\hat y_0
=
a_0^T\hat x.
$$

The variance of the estimated mean is

$$
\operatorname{Var}(\hat y_0\mid A)
=
\sigma^2
a_0^T
(A^TA)^{-1}
a_0.
$$

---

# 63. Prediction Interval vs Confidence Interval

There are two different questions.

## Mean response

What is the mean response at `a_0`?

Uncertainty comes from estimating the coefficients.

## New observation

What value will one new noisy observation take?

Now there is both:

1. coefficient-estimation uncertainty
2. new observation noise

Therefore the prediction variance is

$$
\sigma^2
\left[
1+
a_0^T(A^TA)^{-1}a_0
\right].
$$

The extra `1` is the irreducible noise of the new observation.

Thus prediction intervals are wider than confidence intervals for the mean response.

---

# 64. Leverage

The projection matrix is

$$
P=A(A^TA)^{-1}A^T.
$$

The diagonal element

$$
p_{ii}
$$

is called the leverage of observation `i`.

A point with large leverage has an unusual predictor vector compared with the rest of the data.

Important identity:

$$
\sum_{i=1}^{n}p_{ii}
=
\operatorname{tr}(P)
=
p.
$$

Therefore the average leverage is

$$
\frac{p}{n}.
$$

---

# 65. Residual Variance

Since

$$
e=(I-P)\varepsilon,
$$

we have

$$
\operatorname{Var}(e\mid A)
=
(I-P)
\sigma^2I
(I-P)^T.
$$

Because `I-P` is symmetric and idempotent,

$$
\operatorname{Var}(e\mid A)
=
\sigma^2(I-P).
$$

Hence

$$
\operatorname{Var}(e_i\mid A)
=
\sigma^2(1-p_{ii}).
$$

So even if the original errors have constant variance, fitted residuals do not all have exactly the same variance.

---

# 66. Multicollinearity

Multicollinearity means predictor columns are strongly related.

Perfect multicollinearity means exact dependence:

$$
a_3=c_1a_1+c_2a_2.
$$

Then `A^T A` is singular.

Near multicollinearity means the dependence is approximate.

Then the inverse exists but may be numerically unstable.

Consequences:

- large coefficient standard errors
- unstable coefficients
- large changes after small changes in data
- difficult coefficient interpretation

---

# 67. Omitted-Variable Bias

Suppose the true model is

$$
b=Ax+Cz+\varepsilon,
$$

but we fit only

$$
b\approx Ax.
$$

Then

$$
\hat x
=
(A^TA)^{-1}A^Tb.
$$

Substitute the true model:

$$
\hat x
=
x
+
(A^TA)^{-1}A^TCz
+
(A^TA)^{-1}A^T\varepsilon.
$$

If the omitted variables represented by `Cz` are correlated with the included predictors in `A`, then the middle term is generally nonzero.

This creates omitted-variable bias.

This is one reason why the assumption

$$
E[\varepsilon\mid A]=0
$$

is so important.

---

# 68. Heteroskedasticity

If

$$
\operatorname{Var}(\varepsilon_i\mid A)
$$

changes across observations, the errors are heteroskedastic.

Then the error covariance matrix is more generally

$$
\operatorname{Var}(\varepsilon\mid A)=\Omega.
$$

The OLS coefficient estimator is still

$$
\hat x
=
(A^TA)^{-1}A^Tb.
$$

But its covariance becomes

$$
\operatorname{Var}(\hat x\mid A)
=
(A^TA)^{-1}
A^T
\Omega
A
(A^TA)^{-1}.
$$

So the simple formula

$$
\sigma^2(A^TA)^{-1}
$$

is no longer valid.

---

# 69. Correlated Errors

If

$$
\operatorname{Cov}(\varepsilon_i,\varepsilon_j)\ne0,
$$

the error covariance matrix is not diagonal.

This commonly appears in:

- time series
- panel data
- spatial data
- repeated measurements

The OLS coefficient formula can still be used, but ordinary standard errors may be incorrect.

---

# 70. Outliers and Least Squares

Because OLS minimizes

$$
\sum e_i^2,
$$

large residuals receive large penalties.

For example,

$$
e=2
$$

contributes

$$
4,
$$

whereas

$$
e=10
$$

contributes

$$
100.
$$

Therefore response outliers can strongly influence the fitted regression.

Predictor outliers with high leverage can also strongly affect the coefficients.

---

# 71. Hessian and Convexity of the Least-Squares Objective

Recall

$$
J(x)
=
b^Tb
-
2x^TA^Tb
+
x^TA^TAx.
$$

The gradient is

$$
\nabla J
=
-2A^Tb+2A^TAx.
$$

The Hessian is

$$
\nabla^2J
=
2A^TA.
$$

For any vector `v`,

$$
v^TA^TAv
=
\|Av\|^2
\ge0.
$$

Therefore

$$
A^TA
$$

is positive semidefinite.

So the least-squares objective is convex.

If `A` has full column rank,

$$
A^TA
$$

is positive definite.

Then the objective is strictly convex and the minimizer is unique.

---

# 72. Ridge Regression Connection

Ridge regression modifies the least-squares objective:

$$
J_{ridge}(x)
=
\|b-Ax\|^2
+
\lambda\|x\|^2.
$$

Differentiate:

$$
\nabla J_{ridge}
=
-2A^Tb
+
2A^TAx
+
2\lambda x.
$$

Set equal to zero:

$$
A^TAx+\lambda x=A^Tb.
$$

Therefore

$$
(A^TA+\lambda I)x=A^Tb.
$$

Hence

$$
\hat x_{ridge}
=
(A^TA+\lambda I)^{-1}A^Tb.
$$

Ridge trades some bias for lower variance and improved numerical stability.

---

# 73. Relation Between Simple and Multiple Linear Regression

Simple linear regression is just a special case of the matrix model.

For one predictor,

$$
A=
\begin{bmatrix}
1&t_1\\
1&t_2\\
\vdots&\vdots\\
1&t_n
\end{bmatrix}
$$

and

$$
x=
\begin{bmatrix}
\beta_0\\
\beta_1
\end{bmatrix}.
$$

Then

$$
Ax
=
\begin{bmatrix}
\beta_0+\beta_1t_1\\
\beta_0+\beta_1t_2\\
\vdots\\
\beta_0+\beta_1t_n
\end{bmatrix}.
$$

Thus the same least-squares formula

$$
\hat x
=
(A^TA)^{-1}A^Tb
$$

contains the familiar slope and intercept formulas as a special case.

---

# 74. Interpretation of Coefficients in Multiple Regression

Suppose

$$
y
=
\beta_0
+
\beta_1z_1
+
\beta_2z_2
+
\varepsilon.
$$

Then `beta_1` means:

> Expected change in `y` for a one-unit increase in `z_1`, while holding `z_2` constant.

Similarly `beta_2` is interpreted while holding `z_1` constant.

This "holding other predictors fixed" idea is what distinguishes multiple-regression coefficients from simple pairwise relationships.

---

# 75. Interaction Terms

Suppose

$$
y
=
\beta_0
+
\beta_1z_1
+
\beta_2z_2
+
\beta_3z_1z_2
+
\varepsilon.
$$

Then the effect of `z_1` is

$$
\frac{\partial E[y]}{\partial z_1}
=
\beta_1+\beta_3z_2.
$$

So the effect of one predictor depends on the value of another predictor.

The model is still linear in the parameters.

---

# 76. Polynomial Regression Is Still Linear Regression

Consider

$$
y
=
\beta_0
+
\beta_1z
+
\beta_2z^2
+
\beta_3z^3
+
\varepsilon.
$$

Define feature columns:

$$
a_1=\mathbf 1,
$$

$$
a_2=z,
$$

$$
a_3=z^2,
$$

$$
a_4=z^3.
$$

Then

$$
b=Ax+\varepsilon.
$$

So polynomial regression is simply multiple linear regression with transformed features.

---

# 77. Simple Regression and Correlation

For simple regression with an intercept,

$$
\hat\beta_1
=
r_{ty}
\frac{s_y}{s_t}.
$$

Also,

$$
R^2=r_{ty}^2.
$$

This identity is special to simple linear regression with an intercept.

A strong correlation produces a high simple-regression `R^2`, while a weak correlation produces a low one.

---

# 78. Key Difference Between Regression Coefficients and Correlation

Correlation is symmetric:

$$
r_{xy}=r_{yx}.
$$

Regression is not symmetric.

Predicting `y` from `x` is not the same optimization problem as predicting `x` from `y`.

Regression coefficients also depend on measurement units.

Correlation does not.

---

# 79. Training Error and Generalization

A lower training RSS does not automatically mean better prediction on unseen data.

A model with too many predictors can fit random noise.

This is overfitting.

Therefore model quality should also be evaluated using unseen or held-out data.

Common prediction metrics include:

- Mean Squared Error
- Root Mean Squared Error
- Mean Absolute Error
- `R^2`

---

# 80. Mean Squared Error

For `n` observations,

$$
MSE
=
\frac{1}{n}
\sum_{i=1}^{n}
(y_i-\hat y_i)^2.
$$

So

$$
MSE
=
\frac{RSS}{n}.
$$

Training MSE is an optimization metric.

The unbiased estimator of the model's error variance uses

$$
\frac{RSS}{n-p}
$$

instead.

These two denominators serve different purposes.

---

# 81. Root Mean Squared Error

$$
RMSE
=
\sqrt{
\frac{1}{n}
\sum_{i=1}^{n}
(y_i-\hat y_i)^2
}.
$$

RMSE has the same physical units as the response variable.

---

# 82. Mean Absolute Error

$$
MAE
=
\frac{1}{n}
\sum_{i=1}^{n}
|y_i-\hat y_i|.
$$

Compared with squared error, absolute error is less dominated by very large residuals.

---

# 83. Complete Flow of Linear Regression

The full logic can be remembered as:

1. Start with observations.
2. Build the design matrix `A`.
3. Put unknown coefficients into `x`.
4. Put observed outputs into `b`.
5. Write

$$
b=Ax+\varepsilon.
$$

6. Exact solution usually does not exist.
7. Minimize

$$
\|b-Ax\|^2.
$$

8. At the optimum,

$$
e=b-A\hat x
$$

is perpendicular to the column space.
9. Therefore

$$
A^Te=0.
$$

10. Hence

$$
A^TA\hat x=A^Tb.
$$

11. If `A^T A` is invertible,

$$
\hat x=(A^TA)^{-1}A^Tb.
$$

12. The fitted vector is

$$
\hat b=A\hat x.
$$

13. Equivalently,

$$
\hat b=Pb.
$$

14. The residual is

$$
e=(I-P)b.
$$

15. Add statistical assumptions to study unbiasedness, variance, confidence intervals, and hypothesis tests.

---

# 84. Most Important Formulas

## Model

$$
b=Ax+\varepsilon.
$$

## Prediction

$$
\hat b=A\hat x.
$$

## Residual

$$
e=b-A\hat x.
$$

## Least-squares objective

$$
\hat x
=
\arg\min_x
\|b-Ax\|^2.
$$

## Orthogonality condition

$$
A^Te=0.
$$

## Normal equations

$$
A^TA\hat x=A^Tb.
$$

## OLS solution

$$
\hat x
=
(A^TA)^{-1}A^Tb.
$$

## Projection matrix

$$
P
=
A(A^TA)^{-1}A^T.
$$

## Fitted vector

$$
\hat b=Pb.
$$

## Residual-maker matrix

$$
M=I-P.
$$

## Residual vector

$$
e=Mb.
$$

## OLS unbiasedness

$$
E[\hat x\mid A]=x.
$$

## OLS covariance

$$
\operatorname{Var}(\hat x\mid A)
=
\sigma^2(A^TA)^{-1}.
$$

## Error-variance estimate

$$
s^2
=
\frac{RSS}{n-p}.
$$

## Sum-of-squares decomposition

$$
TSS=ESS+RSS.
$$

## Coefficient of determination

$$
R^2
=
1-\frac{RSS}{TSS}.
$$

---

# 85. Final Mental Picture

There are two equally useful ways to understand linear regression.

## Optimization view

Find the coefficient vector that minimizes squared prediction error:

$$
\hat x
=
\arg\min_x
\|b-Ax\|^2.
$$

## Geometry view

Project the observed target vector `b` onto the column space of `A`:

$$
\hat b
=
P b.
$$

The residual is perpendicular to the model space:

$$
e\perp\operatorname{Col}(A).
$$

Therefore

$$
A^Te=0.
$$

## Linear algebra view

The orthogonality condition produces the normal equations:

$$
A^TA\hat x=A^Tb.
$$

and hence

$$
\hat x
=
(A^TA)^{-1}A^Tb.
$$

## Statistical view

After adding assumptions about the random error:

$$
E[\varepsilon\mid A]=0
$$

gives unbiasedness, while

$$
\operatorname{Var}(\varepsilon\mid A)=\sigma^2I
$$

gives the classical variance formula and Gauss-Markov result.

With Gaussian errors, least squares is also maximum likelihood.

That is the complete connection between:

- simple linear regression
- multiple linear regression
- least squares
- `Ax = b`
- column spaces
- orthogonal projection
- normal equations
- statistical assumptions
- estimator properties
- inference
- prediction
