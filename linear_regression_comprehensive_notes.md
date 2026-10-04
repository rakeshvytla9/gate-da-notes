# Linear Regression — Comprehensive Notes with Proofs, Geometry, Linear Algebra, and Statistical Assumptions

> **Purpose of these notes.**  
> These notes use the uploaded PDF *Machine Learning Problem Types Explained* as the starting point, especially its discussion of OLS geometry, column spaces, normal equations, projection matrices, and the \(Ax=b\) viewpoint.  
> I have also **checked the mathematical claims** in the PDF. Wherever a statement needs qualification or correction, it is explicitly marked as a **PDF check / correction** rather than silently changed.

---

## 0. What Linear Regression Is

Linear regression models a continuous response \(y\) as a linear function of one or more predictors.

For observation \(i\),

\[
y_i = \beta_0+\beta_1x_{i1}+\beta_2x_{i2}+\cdots+\beta_{p-1}x_{i,p-1}+\varepsilon_i.
\]

The model is **linear in the unknown parameters** \(\beta_j\).  
The predictors themselves do not have to appear only as raw variables. For example,

\[
y_i=\beta_0+\beta_1x_i+\beta_2x_i^2+\varepsilon_i
\]

is still a linear regression model because it is linear in \(\beta_0,\beta_1,\beta_2\).

Linear regression is a regression method because the target is continuous, unlike classification where the target is categorical.

---

# Part I — Notation and Matrix Form

## 1. Samples, Features, and Parameters

Let

- \(n\) = number of observations / samples,
- \(p\) = number of regression parameters,
- \(X\in\mathbb R^{n\times p}\) = design matrix,
- \(y\in\mathbb R^n\) = observed response vector,
- \(\beta\in\mathbb R^p\) = unknown coefficient vector,
- \(\varepsilon\in\mathbb R^n\) = random error vector.

Then

\[
y=X\beta+\varepsilon.
\]

If the model contains an intercept, the first column of \(X\) is usually a vector of ones:

\[
X=
\begin{bmatrix}
1 & x_{11} & x_{12} & \cdots \\
1 & x_{21} & x_{22} & \cdots \\
\vdots & \vdots & \vdots & \\
1 & x_{n1} & x_{n2} & \cdots
\end{bmatrix}.
\]

The response vector is

\[
y=
\begin{bmatrix}
y_1\\
y_2\\
\vdots\\
y_n
\end{bmatrix}.
\]

The parameter vector is

\[
\beta=
\begin{bmatrix}
\beta_0\\
\beta_1\\
\vdots\\
\beta_{p-1}
\end{bmatrix}.
\]

---

## 2. Why the Observation Space Has \(n\) Dimensions

Each feature column contains one number for each observation:

\[
x_j=
\begin{bmatrix}
x_{1j}\\
x_{2j}\\
\vdots\\
x_{nj}
\end{bmatrix}
\in\mathbb R^n.
\]

Likewise,

\[
y\in\mathbb R^n.
\]

Therefore, in the **column-space geometric view**, each coordinate axis corresponds to an observation index.

For \(n=3\),

\[
x_j=
\begin{bmatrix}
x_{1j}\\
x_{2j}\\
x_{3j}
\end{bmatrix}
\]

can be drawn as one arrow in ordinary 3D space.

For \(n=1000\), the same idea lives in \(\mathbb R^{1000}\).

### Important conceptual correction

It is tempting to say that these axes are orthogonal *because the observations are statistically independent*. That is not the real reason.

The axes of \(\mathbb R^n\) are orthogonal because we **define the usual Euclidean coordinate system that way**. Statistical independence of observations is a separate probabilistic assumption.

So:

- geometric orthogonality = property of the vector space,
- statistical independence = property of random variables.

They are not the same idea.

---

## 3. Column Space of \(X\)

Write the design matrix by columns:

\[
X=
\begin{bmatrix}
| & | & & |\\
x_1 & x_2 & \cdots & x_p\\
| & | & & |
\end{bmatrix}.
\]

Then

\[
X\beta
=
\beta_1x_1+\beta_2x_2+\cdots+\beta_px_p.
\]

Therefore every possible fitted response lies in

\[
\operatorname{Col}(X)
=
\operatorname{span}\{x_1,x_2,\ldots,x_p\}.
\]

So the regression model can only produce vectors inside the column space of \(X\).

If the columns are linearly independent,

\[
\dim\operatorname{Col}(X)=p.
\]

More generally,

\[
\dim\operatorname{Col}(X)
=
\operatorname{rank}(X)
\le \min(n,p).
\]

---

# Part II — OLS as an Approximate Solution of \(Ax=b\)

## 4. The Exact System

A linear system

\[
Ax=b
\]

has an exact solution only if

\[
b\in\operatorname{Col}(A).
\]

In regression, identify

\[
A=X,\qquad x=\beta,\qquad b=y.
\]

Then exact fitting means

\[
X\beta=y.
\]

Usually, especially when \(n>p\), there are more equations than unknowns and noisy data make exact equality impossible.

So OLS solves the approximation problem

\[
\hat\beta
=
\arg\min_\beta
\|y-X\beta\|_2^2.
\]

Define the fitted vector

\[
\hat y=X\hat\beta
\]

and residual vector

\[
e=y-\hat y=y-X\hat\beta.
\]

The OLS objective is

\[
S(\beta)
=
\sum_{i=1}^n e_i^2
=
\|y-X\beta\|_2^2.
\]

---

# Part III — Scalar Derivation for Simple Linear Regression

## 5. Simple Linear Regression

Suppose

\[
y_i=\beta_0+\beta_1x_i+\varepsilon_i.
\]

OLS chooses \(b_0,b_1\) to minimize

\[
S(b_0,b_1)
=
\sum_{i=1}^n
(y_i-b_0-b_1x_i)^2.
\]

Take partial derivatives.

### Derivative with respect to \(b_0\)

\[
\frac{\partial S}{\partial b_0}
=
-2\sum_{i=1}^n
(y_i-b_0-b_1x_i).
\]

Set equal to zero:

\[
\sum_{i=1}^n
(y_i-b_0-b_1x_i)=0.
\]

Therefore

\[
\sum y_i
=
nb_0+b_1\sum x_i.
\]

Divide by \(n\):

\[
\bar y=b_0+b_1\bar x.
\]

Hence

\[
\boxed{
b_0=\bar y-b_1\bar x
}
\]

---

## 6. Derivative with respect to \(b_1\)

\[
\frac{\partial S}{\partial b_1}
=
-2\sum_{i=1}^n
x_i(y_i-b_0-b_1x_i).
\]

Set equal to zero:

\[
\sum x_i(y_i-b_0-b_1x_i)=0.
\]

Substitute

\[
b_0=\bar y-b_1\bar x.
\]

Then

\[
y_i-b_0-b_1x_i
=
(y_i-\bar y)-b_1(x_i-\bar x).
\]

Therefore,

\[
\sum x_i
\left[
(y_i-\bar y)-b_1(x_i-\bar x)
\right]
=0.
\]

It is cleaner to center \(x_i\):

\[
\sum (x_i-\bar x)(y_i-\bar y)
-
b_1
\sum (x_i-\bar x)^2
=0.
\]

Thus

\[
\boxed{
b_1
=
\frac{
\sum_{i=1}^n
(x_i-\bar x)(y_i-\bar y)
}{
\sum_{i=1}^n
(x_i-\bar x)^2
}
}
\]

and

\[
\boxed{
b_0=\bar y-b_1\bar x.
}
\]

Define

\[
S_{xx}
=
\sum (x_i-\bar x)^2,
\]

\[
S_{xy}
=
\sum (x_i-\bar x)(y_i-\bar y).
\]

Then

\[
\boxed{
b_1=\frac{S_{xy}}{S_{xx}}.
}
\]

This shows that the slope is essentially a normalized covariance.

---

# Part IV — Matrix Calculus Derivation of OLS

## 7. OLS Objective

We minimize

\[
S(\beta)
=
\|y-X\beta\|_2^2.
\]

Since

\[
\|v\|_2^2=v^Tv,
\]

we have

\[
S(\beta)
=
(y-X\beta)^T(y-X\beta).
\]

Expand:

\[
S(\beta)
=
y^Ty-y^TX\beta-\beta^TX^Ty+\beta^TX^TX\beta.
\]

Because \(y^TX\beta\) is a scalar,

\[
y^TX\beta
=
(y^TX\beta)^T
=
\beta^TX^Ty.
\]

Therefore

\[
S(\beta)
=
y^Ty
-
2\beta^TX^Ty
+
\beta^TX^TX\beta.
\]

---

## 8. Gradient

Use

\[
\nabla_\beta(\beta^Ta)=a
\]

and, for symmetric \(A\),

\[
\nabla_\beta(\beta^TA\beta)=2A\beta.
\]

Since \(X^TX\) is symmetric,

\[
\nabla_\beta S(\beta)
=
-2X^Ty+2X^TX\beta.
\]

At the minimizer,

\[
\nabla_\beta S(\hat\beta)=0.
\]

Therefore

\[
-2X^Ty+2X^TX\hat\beta=0.
\]

Hence

\[
\boxed{
X^TX\hat\beta=X^Ty
}
\]

which are the **normal equations**.

If \(X\) has full column rank,

\[
\operatorname{rank}(X)=p,
\]

then \(X^TX\) is invertible and

\[
\boxed{
\hat\beta
=
(X^TX)^{-1}X^Ty.
}
\]

---

# Part V — Why \(X^TX\) Is Invertible Under Full Column Rank

## 9. Proof

Suppose \(X\) has full column rank.

For any nonzero \(v\in\mathbb R^p\),

\[
v^TX^TXv
=
(Xv)^T(Xv)
=
\|Xv\|_2^2.
\]

Since \(X\) has full column rank,

\[
Xv=0
\quad\Longrightarrow\quad
v=0.
\]

Thus for every nonzero \(v\),

\[
\|Xv\|_2^2>0.
\]

Hence

\[
v^TX^TXv>0.
\]

Therefore \(X^TX\) is **positive definite**.

Every positive definite matrix is nonsingular, so

\[
(X^TX)^{-1}
\]

exists.

---

# Part VI — Geometric Proof of OLS

## 10. The Main Geometric Picture

The predicted vector

\[
\hat y=X\hat\beta
\]

must lie inside

\[
\operatorname{Col}(X).
\]

The observed response vector \(y\) is usually outside that subspace.

OLS asks:

> Which vector in \(\operatorname{Col}(X)\) is closest to \(y\) in Euclidean distance?

The answer is the **orthogonal projection** of \(y\) onto \(\operatorname{Col}(X)\).

So the residual

\[
e=y-\hat y
\]

must be perpendicular to the entire column space.

Thus

\[
e\perp\operatorname{Col}(X).
\]

Since every column \(x_j\) belongs to \(\operatorname{Col}(X)\),

\[
x_j^Te=0
\qquad
\text{for every }j.
\]

Stacking all \(p\) conditions gives

\[
\boxed{
X^Te=0.
}
\]

Since

\[
e=y-X\hat\beta,
\]

we get

\[
X^T(y-X\hat\beta)=0.
\]

Expand:

\[
X^Ty-X^TX\hat\beta=0.
\]

Therefore

\[
\boxed{
X^TX\hat\beta=X^Ty.
}
\]

So the normal equations arise directly from orthogonal projection geometry.

---

## 11. Why Orthogonality Gives the Nearest Point

Let \(s\in\operatorname{Col}(X)\) be any other candidate prediction.

Because \(\hat y\in\operatorname{Col}(X)\),

\[
s-\hat y\in\operatorname{Col}(X).
\]

The OLS residual satisfies

\[
e=y-\hat y
\perp
\operatorname{Col}(X).
\]

Hence

\[
e\perp(s-\hat y).
\]

Now write

\[
y-s
=
(y-\hat y)+(\hat y-s).
\]

The two components are perpendicular, so by Pythagoras,

\[
\|y-s\|^2
=
\|y-\hat y\|^2
+
\|\hat y-s\|^2.
\]

Since the second term is nonnegative,

\[
\|y-s\|^2
\ge
\|y-\hat y\|^2.
\]

Therefore \(\hat y\) is indeed the closest vector in the subspace.

This is a complete geometric proof of least squares optimality.

---

# Part VII — A Concrete 3D Visualization

## 12. One Feature, Three Samples

Suppose

\[
x=
\begin{bmatrix}
1\\
2\\
3
\end{bmatrix},
\qquad
y=
\begin{bmatrix}
2\\
1\\
4
\end{bmatrix}.
\]

Without an intercept,

\[
\hat y=\beta x.
\]

All possible predictions are scalar multiples of \(x\).

So they lie on the line

\[
\operatorname{span}\{x\}
\]

inside \(\mathbb R^3\).

The OLS coefficient is

\[
\hat\beta
=
\frac{x^Ty}{x^Tx}.
\]

Compute:

\[
x^Ty
=
1(2)+2(1)+3(4)=16,
\]

\[
x^Tx
=
1^2+2^2+3^2=14.
\]

Therefore

\[
\hat\beta
=
\frac{16}{14}
=
\frac87.
\]

Then

\[
\hat y
=
\frac87
\begin{bmatrix}
1\\
2\\
3
\end{bmatrix}
=
\begin{bmatrix}
8/7\\
16/7\\
24/7
\end{bmatrix}.
\]

Residual:

\[
e
=
y-\hat y
=
\begin{bmatrix}
2\\
1\\
4
\end{bmatrix}
-
\begin{bmatrix}
8/7\\
16/7\\
24/7
\end{bmatrix}
=
\begin{bmatrix}
6/7\\
-9/7\\
4/7
\end{bmatrix}.
\]

Check orthogonality:

\[
x^Te
=
1\frac67
+
2\left(-\frac97\right)
+
3\frac47
=
\frac{6-18+12}{7}
=0.
\]

Thus the residual is perpendicular to the line spanned by \(x\).

---

# Part VIII — Multiple Features as a Plane or Hyperplane

## 13. Two Features, Three Samples

Suppose

\[
X=
\begin{bmatrix}
| & |\\
x_1 & x_2\\
| & |
\end{bmatrix}
\in\mathbb R^{3\times2}.
\]

If \(x_1\) and \(x_2\) are linearly independent, then

\[
\operatorname{Col}(X)
=
\operatorname{span}\{x_1,x_2\}
\]

is a 2D plane through the origin inside 3D space.

Every prediction is

\[
\hat y
=
\beta_1x_1+\beta_2x_2.
\]

So changing \(\beta_1,\beta_2\) moves the predicted point around that plane.

The observed \(y\) is generally off the plane.

OLS drops a perpendicular from \(y\) onto the plane.

The perpendicular residual satisfies

\[
x_1^Te=0,
\]

\[
x_2^Te=0.
\]

Together,

\[
X^Te=0.
\]

For general \(n\) and \(p\), exactly the same geometry holds in higher-dimensional space.

---

# Part IX — Data-Space Geometry vs Column-Space Geometry

## 14. Why the Residual Is Not Usually Perpendicular to the Regression Line in an \(x\)-\(y\) Scatterplot

This is one of the most important conceptual distinctions.

### In the ordinary \(x\)-\(y\) plot

For simple regression,

\[
\hat y_i=b_0+b_1x_i.
\]

The residual is

\[
e_i=y_i-\hat y_i.
\]

OLS minimizes

\[
\sum_i
(y_i-\hat y_i)^2.
\]

This is a sum of **vertical** squared distances.

It does **not** minimize the perpendicular Euclidean distances from each point to the fitted line.

Therefore, in the ordinary scatterplot, the residual segment from a point to the regression line is vertical, not generally perpendicular to the line.

### In \(\mathbb R^n\) column-space geometry

There is only one response vector

\[
y=
(y_1,\ldots,y_n)^T.
\]

OLS projects this vector orthogonally onto

\[
\operatorname{Col}(X).
\]

Therefore

\[
e=y-\hat y
\]

is genuinely perpendicular to the feature subspace.

These are different geometric spaces.

---

## 15. OLS vs Total Least Squares

OLS treats the response direction specially:

\[
y=X\beta+\varepsilon.
\]

The classical regression formulation attributes stochastic error to the response conditional on \(X\).

Total least squares (TLS), orthogonal regression, or some errors-in-variables models instead allow deviations in both predictor and response coordinates.

For a simple line

\[
y=\alpha+\beta x,
\]

the perpendicular distance from \((x_i,y_i)\) to the line is

\[
d_i
=
\frac{
|y_i-\alpha-\beta x_i|
}{
\sqrt{1+\beta^2}
}.
\]

TLS minimizes perpendicular distances, not vertical residuals.

### Important qualification

Ordinary Euclidean TLS depends on the relative scaling of coordinates.

If \(x\) is measured in years and \(y\) in rupees, Euclidean distance in the raw coordinate system has no obvious physical meaning.

This does **not** mean TLS is impossible whenever units differ. More general weighted or generalized TLS methods can incorporate measurement-error covariance or rescaling.

---

# Part X — Projection Matrix / Hat Matrix

## 16. Definition

If \(X\) has full column rank,

\[
\hat\beta
=
(X^TX)^{-1}X^Ty.
\]

Therefore

\[
\hat y
=
X\hat\beta
=
X(X^TX)^{-1}X^Ty.
\]

Define

\[
\boxed{
H
=
X(X^TX)^{-1}X^T.
}
\]

Then

\[
\boxed{
\hat y=Hy.
}
\]

\(H\) is called the **hat matrix** because it puts a “hat” on \(y\).

---

## 17. Symmetry of \(H\)

\[
H^T
=
\left[
X(X^TX)^{-1}X^T
\right]^T.
\]

Using \((ABC)^T=C^TB^TA^T\),

\[
H^T
=
X
\left[
(X^TX)^{-1}
\right]^T
X^T.
\]

Since \(X^TX\) is symmetric, its inverse is symmetric:

\[
\left[
(X^TX)^{-1}
\right]^T
=
(X^TX)^{-1}.
\]

Thus

\[
\boxed{
H^T=H.
}
\]

---

## 18. Idempotence of \(H\)

\[
H^2
=
X(X^TX)^{-1}X^T
X(X^TX)^{-1}X^T.
\]

Regroup:

\[
H^2
=
X(X^TX)^{-1}
(X^TX)
(X^TX)^{-1}X^T.
\]

Hence

\[
H^2
=
X(X^TX)^{-1}X^T
=
H.
\]

Therefore

\[
\boxed{
H^2=H.
}
\]

Interpretation: once a vector has been projected into the column space, projecting it again does nothing.

---

## 19. Residual-Maker Matrix

Define

\[
M=I-H.
\]

Then

\[
e=y-\hat y
=
y-Hy
=
(I-H)y
=
My.
\]

\(M\) projects onto the orthogonal complement of \(\operatorname{Col}(X)\).

Properties:

\[
M^T=M,
\]

\[
M^2=M,
\]

\[
HX=X,
\]

\[
MX=0,
\]

\[
HM=0.
\]

---

# Part XI — Important Consequences of OLS Orthogonality

## 20. Residuals Are Orthogonal to Every Regressor

\[
\boxed{
X^Te=0.
}
\]

Equivalently, for every regressor \(x_j\),

\[
x_j^Te=0.
\]

---

## 21. Residuals Are Orthogonal to Fitted Values

Because

\[
\hat y=X\hat\beta,
\]

we get

\[
\hat y^Te
=
\hat\beta^TX^Te.
\]

But

\[
X^Te=0.
\]

Therefore

\[
\boxed{
\hat y^Te=0.
}
\]

---

## 22. Residuals Sum to Zero When an Intercept Is Included

If \(X\) contains a column of ones

\[
\mathbf 1=
\begin{bmatrix}
1\\
\vdots\\
1
\end{bmatrix},
\]

then orthogonality gives

\[
\mathbf 1^Te=0.
\]

Thus

\[
\sum_{i=1}^n e_i=0.
\]

Hence

\[
\boxed{
\bar e=0.
}
\]

This property generally fails if the model does not include an intercept.

---

## 23. Mean Fitted Value Equals Mean Observed Value

Since

\[
e_i=y_i-\hat y_i
\]

and

\[
\sum_i e_i=0,
\]

we obtain

\[
\sum_i y_i
=
\sum_i\hat y_i.
\]

Thus

\[
\boxed{
\bar{\hat y}=\bar y.
}
\]

Again, this relies on including an intercept.

---

# Part XII — Orthogonal Decomposition of the Observation Space

## 24. Fundamental Decomposition

For any matrix \(X\),

\[
\mathbb R^n
=
\operatorname{Col}(X)
\oplus
\operatorname{Null}(X^T).
\]

This means every vector \(y\in\mathbb R^n\) can be written uniquely as

\[
y
=
y_{\parallel}
+
y_{\perp},
\]

where

\[
y_{\parallel}
\in
\operatorname{Col}(X),
\]

and

\[
y_{\perp}
\in
\operatorname{Null}(X^T).
\]

For OLS,

\[
y_{\parallel}=\hat y,
\]

\[
y_{\perp}=e.
\]

So

\[
\boxed{
y=\hat y+e.
}
\]

and

\[
\boxed{
\hat y\perp e.
}
\]

---

## 25. Dimensions of the Two Subspaces

Let

\[
r=\operatorname{rank}(X).
\]

Then

\[
\dim\operatorname{Col}(X)=r,
\]

and by the rank-nullity theorem applied to \(X^T\),

\[
\dim\operatorname{Null}(X^T)
=
n-r.
\]

Thus

\[
r+(n-r)=n.
\]

If \(X\) has full column rank \(p\),

\[
r=p,
\]

so

\[
\dim\operatorname{Null}(X^T)
=
n-p.
\]

These \(n-p\) dimensions are directions in response space that the linear model cannot reproduce.

---

# Part XIII — Rank Cases and When Exact Fitting Is Possible

## 26. Case 1: \(n>p\) — Overdetermined System

There are more observations than parameters.

If \(X\) has full column rank,

\[
\operatorname{rank}(X)=p.
\]

Then

\[
\dim\operatorname{Col}(X)=p<n.
\]

So the model subspace occupies only a lower-dimensional part of \(\mathbb R^n\).

A generic response \(y\) will not lie exactly in that subspace.

Therefore

\[
X\beta=y
\]

usually has no exact solution, and OLS finds the nearest point.

---

## 27. Case 2: \(n=p\)

A square design matrix does **not** automatically imply exact interpolation.

Exact interpolation for every \(y\) requires

\[
\operatorname{rank}(X)=n=p.
\]

Equivalently, \(X\) must be invertible.

Then

\[
\beta=X^{-1}y
\]

and

\[
\hat y=y.
\]

But if \(X\) is singular,

\[
\operatorname{rank}(X)<n,
\]

then

\[
\operatorname{Col}(X)\ne\mathbb R^n,
\]

so some response vectors cannot be fitted exactly.

### PDF check / correction

Any claim of the form

> “if \(p=n\), the columns span all of \(\mathbb R^n\)”

needs the extra condition

\[
\operatorname{rank}(X)=n.
\]

---

## 28. Case 3: \(n<p\) — Underdetermined / High-Dimensional System

Now there are more parameters than observations.

The rank satisfies

\[
\operatorname{rank}(X)\le n<p.
\]

If \(X\) has **full row rank**,

\[
\operatorname{rank}(X)=n,
\]

then

\[
\operatorname{Col}(X)=\mathbb R^n.
\]

Therefore every \(y\in\mathbb R^n\) can be represented exactly:

\[
X\beta=y.
\]

Because there are more unknowns than independent equations, there are infinitely many exact solutions.

However, if

\[
\operatorname{rank}(X)<n,
\]

then

\[
\operatorname{Col}(X)\ne\mathbb R^n,
\]

so an arbitrary \(y\) may still not be exactly representable.

### PDF check / correction

Thus

> “\(n<p\) implies infinitely many zero-training-error solutions”

is only guaranteed when

\[
\operatorname{rank}(X)=n.
\]

---

# Part XIV — Rank Deficiency and the Pseudoinverse

## 29. When \(X^TX\) Is Singular

If the columns of \(X\) are linearly dependent, then

\[
\operatorname{rank}(X)<p.
\]

Then there exists some nonzero \(v\) such that

\[
Xv=0.
\]

Consequently,

\[
v^TX^TXv
=
\|Xv\|^2
=
0,
\]

so \(X^TX\) cannot be positive definite and is singular.

The coefficient vector minimizing SSE may not be unique.

However, the fitted vector \(\hat y\), i.e. the projection of \(y\) onto \(\operatorname{Col}(X)\), is still unique.

---

## 30. Moore–Penrose Pseudoinverse

The general minimum-norm least-squares solution is

\[
\boxed{
\hat\beta=X^+y
}
\]

where \(X^+\) is the Moore–Penrose pseudoinverse.

If \(X\) has full column rank,

\[
X^+
=
(X^TX)^{-1}X^T.
\]

If \(X\) has full row rank,

\[
X^+
=
X^T(XX^T)^{-1}.
\]

The projection matrix can be written generally as

\[
H=XX^+.
\]

---

# Part XV — Classical Linear Model Assumptions

The geometry of OLS requires far fewer assumptions than statistical inference.

It is important to separate:

1. assumptions needed to **compute** OLS,
2. assumptions needed for **unbiasedness**,
3. assumptions needed for **Gauss–Markov efficiency**,
4. assumptions needed for exact **\(t\)- and \(F\)-inference**.

---

## 31. Assumption A1 — Linear in Parameters

The model is

\[
y=X\beta+\varepsilon.
\]

This means the conditional mean is linear in the coefficients:

\[
E[y\mid X]=X\beta.
\]

This does not forbid transformations such as \(x^2\), \(\log x\), interactions, splines, or other basis functions.

---

## 32. Assumption A2 — No Perfect Multicollinearity

For the usual unique closed-form estimator,

\[
\operatorname{rank}(X)=p.
\]

Equivalently, no column of \(X\) is an exact linear combination of the others.

Then

\[
X^TX
\]

is invertible.

Approximate multicollinearity does not make OLS undefined, but it may make estimates unstable and their variances large.

---

## 33. Assumption A3 — Zero Conditional Mean / Exogeneity

The key unbiasedness assumption is

\[
\boxed{
E[\varepsilon\mid X]=0.
}
\]

This says the errors have no systematic component predictable from the regressors.

Violations can result from:

- omitted relevant variables correlated with included predictors,
- simultaneity,
- reverse causality,
- measurement error in regressors,
- selection bias,
- certain forms of model misspecification.

This assumption is more important for unbiasedness than normality.

---

## 34. Assumption A4 — Homoskedasticity

The classical Gauss–Markov model assumes

\[
\operatorname{Var}(\varepsilon\mid X)
=
\sigma^2I_n.
\]

This means

\[
\operatorname{Var}(\varepsilon_i\mid X)=\sigma^2
\]

for every observation and

\[
\operatorname{Cov}(\varepsilon_i,\varepsilon_j\mid X)=0
\qquad
i\ne j.
\]

Homoskedasticity is **not required for OLS unbiasedness**.

It is required for the usual simple variance formula

\[
\operatorname{Var}(\hat\beta\mid X)
=
\sigma^2(X^TX)^{-1}
\]

and for the standard Gauss–Markov BLUE result in its classical form.

---

## 35. Assumption A5 — Independent / Uncorrelated Errors

A common classroom assumption is that observations are independently sampled.

Under conditional homoskedasticity, this is often summarized through

\[
\operatorname{Var}(\varepsilon\mid X)=\sigma^2I.
\]

For time series or clustered data, errors can be correlated:

\[
\operatorname{Cov}(\varepsilon_i,\varepsilon_j\mid X)\ne0.
\]

OLS coefficients can still be unbiased under exogeneity, but ordinary standard errors become wrong.

---

## 36. Assumption A6 — Normality

For exact finite-sample classical inference, assume

\[
\varepsilon\mid X
\sim
\mathcal N(0,\sigma^2I).
\]

Then

\[
y\mid X
\sim
\mathcal N(X\beta,\sigma^2I).
\]

Normality is **not required** to:

- define OLS,
- derive the normal equations,
- obtain the geometric projection,
- prove unbiasedness,
- prove Gauss–Markov under the other assumptions.

Normality is mainly used for exact finite-sample \(t\), \(F\), and likelihood results.

---

## 37. Fixed-\(X\) vs Random-\(X\)

Two common viewpoints are:

### Fixed design

Treat \(X\) as nonrandom and condition on it.

### Random design

Treat rows of \(X\) as random observations.

Then results are often stated conditionally on \(X\), e.g.

\[
E[\hat\beta\mid X]=\beta.
\]

Conditioning makes the algebra nearly identical to the fixed-design case.

---

# Part XVI — Unbiasedness of OLS

## 38. Theorem

Suppose

\[
y=X\beta+\varepsilon,
\]

\(X\) has full column rank, and

\[
E[\varepsilon\mid X]=0.
\]

Then

\[
\boxed{
E[\hat\beta\mid X]=\beta.
}
\]

### Proof

Start with

\[
\hat\beta
=
(X^TX)^{-1}X^Ty.
\]

Substitute

\[
y=X\beta+\varepsilon.
\]

Then

\[
\hat\beta
=
(X^TX)^{-1}X^T(X\beta+\varepsilon).
\]

Expand:

\[
\hat\beta
=
(X^TX)^{-1}X^TX\beta
+
(X^TX)^{-1}X^T\varepsilon.
\]

Since

\[
(X^TX)^{-1}X^TX=I,
\]

\[
\hat\beta
=
\beta
+
(X^TX)^{-1}X^T\varepsilon.
\]

Take conditional expectation:

\[
E[\hat\beta\mid X]
=
\beta
+
(X^TX)^{-1}X^T
E[\varepsilon\mid X].
\]

Using

\[
E[\varepsilon\mid X]=0,
\]

we obtain

\[
\boxed{
E[\hat\beta\mid X]=\beta.
}
\]

---

# Part XVII — Variance of OLS

## 39. Derivation Under Homoskedasticity

From

\[
\hat\beta-\beta
=
(X^TX)^{-1}X^T\varepsilon,
\]

let

\[
A=(X^TX)^{-1}X^T.
\]

Then

\[
\hat\beta-\beta=A\varepsilon.
\]

Therefore

\[
\operatorname{Var}(\hat\beta\mid X)
=
A\operatorname{Var}(\varepsilon\mid X)A^T.
\]

Assume

\[
\operatorname{Var}(\varepsilon\mid X)=\sigma^2I.
\]

Then

\[
\operatorname{Var}(\hat\beta\mid X)
=
\sigma^2AA^T.
\]

Now

\[
AA^T
=
(X^TX)^{-1}X^T
X
(X^TX)^{-1}.
\]

Therefore

\[
AA^T
=
(X^TX)^{-1}.
\]

Hence

\[
\boxed{
\operatorname{Var}(\hat\beta\mid X)
=
\sigma^2(X^TX)^{-1}.
}
\]

---

# Part XVIII — Heteroskedastic Case

## 40. General Covariance Matrix

If

\[
\operatorname{Var}(\varepsilon\mid X)=\Omega,
\]

then

\[
\boxed{
\operatorname{Var}(\hat\beta\mid X)
=
(X^TX)^{-1}X^T
\Omega
X(X^TX)^{-1}.
}
\]

The OLS coefficient formula is unchanged, but the classical standard-error formula is no longer valid unless \(\Omega=\sigma^2I\).

This motivates heteroskedasticity-robust standard errors.

---

# Part XIX — Gauss–Markov Theorem

## 41. Statement

Under the classical assumptions

\[
y=X\beta+\varepsilon,
\]

\[
E[\varepsilon\mid X]=0,
\]

\[
\operatorname{Var}(\varepsilon\mid X)=\sigma^2I,
\]

and full column rank of \(X\),

the OLS estimator

\[
\hat\beta=(X^TX)^{-1}X^Ty
\]

is **BLUE**:

> Best Linear Unbiased Estimator.

“Best” means minimum covariance among all linear unbiased estimators.

---

## 42. Proof

Let another estimator be linear in \(y\):

\[
\tilde\beta=Cy
\]

for some matrix \(C\in\mathbb R^{p\times n}\).

For unbiasedness,

\[
E[\tilde\beta\mid X]
=
CE[y\mid X]
=
CX\beta.
\]

For this to equal \(\beta\) for every \(\beta\),

\[
CX=I_p.
\]

Define the OLS linear operator

\[
A=(X^TX)^{-1}X^T.
\]

Then

\[
AX=I_p.
\]

Write

\[
C=A+D.
\]

Since both \(CX=I\) and \(AX=I\),

\[
DX=0.
\]

Now

\[
\tilde\beta
=
(A+D)y.
\]

Its conditional variance is

\[
\operatorname{Var}(\tilde\beta\mid X)
=
\sigma^2(A+D)(A+D)^T.
\]

Expand:

\[
=
\sigma^2
\left(
AA^T+AD^T+DA^T+DD^T
\right).
\]

We show the cross terms vanish.

Since

\[
A=(X^TX)^{-1}X^T,
\]

\[
AD^T
=
(X^TX)^{-1}X^TD^T
=
(X^TX)^{-1}(DX)^T
=
0.
\]

Similarly,

\[
DA^T=0.
\]

Therefore

\[
\operatorname{Var}(\tilde\beta\mid X)
=
\sigma^2AA^T
+
\sigma^2DD^T.
\]

But

\[
\sigma^2AA^T
=
\operatorname{Var}(\hat\beta\mid X).
\]

Hence

\[
\boxed{
\operatorname{Var}(\tilde\beta\mid X)
-
\operatorname{Var}(\hat\beta\mid X)
=
\sigma^2DD^T.
}
\]

For every vector \(z\),

\[
z^TDD^Tz
=
\|D^Tz\|^2
\ge0.
\]

Thus \(DD^T\) is positive semidefinite.

Therefore no other linear unbiased estimator has smaller covariance than OLS.

So OLS is BLUE.

---

# Part XX — Distribution of OLS Under Gaussian Errors

## 43. Conditional Distribution

If

\[
\varepsilon\mid X
\sim
\mathcal N(0,\sigma^2I),
\]

then

\[
\hat\beta
=
\beta
+
(X^TX)^{-1}X^T\varepsilon.
\]

A linear transformation of a Gaussian vector is Gaussian.

Therefore

\[
\boxed{
\hat\beta\mid X
\sim
\mathcal N
\left(
\beta,
\sigma^2(X^TX)^{-1}
\right).
}
\]

For coefficient \(j\),

\[
\hat\beta_j
\sim
\mathcal N
\left(
\beta_j,
\sigma^2[(X^TX)^{-1}]_{jj}
\right).
\]

---

# Part XXI — MLE Derivation Under Gaussian Noise

## 44. Likelihood

Assume

\[
y\mid X
\sim
\mathcal N(X\beta,\sigma^2I).
\]

Then

\[
f(y\mid X,\beta,\sigma^2)
=
(2\pi\sigma^2)^{-n/2}
\exp
\left[
-\frac1{2\sigma^2}
(y-X\beta)^T(y-X\beta)
\right].
\]

The log-likelihood is

\[
\ell(\beta,\sigma^2)
=
-\frac n2\log(2\pi)
-\frac n2\log\sigma^2
-\frac1{2\sigma^2}
\|y-X\beta\|^2.
\]

For fixed \(\sigma^2\), maximizing \(\ell\) with respect to \(\beta\) is equivalent to minimizing

\[
\|y-X\beta\|^2.
\]

Therefore

\[
\boxed{
\hat\beta_{\text{MLE}}
=
\hat\beta_{\text{OLS}}.
}
\]

So OLS and Gaussian maximum likelihood give the same coefficient estimator.

---

## 45. MLE of \(\sigma^2\)

Differentiate the log-likelihood with respect to \(\sigma^2\).

The MLE is

\[
\boxed{
\hat\sigma^2_{\text{MLE}}
=
\frac{\operatorname{SSE}}{n}.
}
\]

This estimator is biased downward.

The usual unbiased estimator is

\[
\boxed{
s^2
=
\frac{\operatorname{SSE}}{n-p}.
}
\]

---

# Part XXII — Why \(n-p\) Appears in the Variance Estimate

## 46. Degrees of Freedom

The residual vector is

\[
e=My
\]

where

\[
M=I-H.
\]

If \(X\) has rank \(p\),

\[
\operatorname{rank}(H)=p.
\]

Therefore

\[
\operatorname{rank}(M)=n-p.
\]

So residuals live in an \((n-p)\)-dimensional subspace.

This is why the residual sum of squares has \(n-p\) residual degrees of freedom.

---

## 47. Unbiasedness of \(s^2\)

Under the classical model,

\[
e=M\varepsilon
\]

because

\[
MX=0.
\]

Then

\[
\operatorname{SSE}
=
e^Te
=
\varepsilon^TM\varepsilon.
\]

Use the identity

\[
E[\varepsilon^TA\varepsilon]
=
\operatorname{tr}
\left(
A\operatorname{Var}(\varepsilon)
\right)
+
E[\varepsilon]^TAE[\varepsilon].
\]

Since

\[
E[\varepsilon]=0
\]

and

\[
\operatorname{Var}(\varepsilon)=\sigma^2I,
\]

\[
E[\operatorname{SSE}\mid X]
=
\sigma^2\operatorname{tr}(M).
\]

For an idempotent projection matrix, trace equals rank:

\[
\operatorname{tr}(M)=n-p.
\]

Hence

\[
E[\operatorname{SSE}\mid X]
=
\sigma^2(n-p).
\]

Therefore

\[
E\left[
\frac{\operatorname{SSE}}{n-p}
\middle|X
\right]
=
\sigma^2.
\]

Thus

\[
\boxed{
s^2=\frac{\operatorname{SSE}}{n-p}
}
\]

is unbiased for \(\sigma^2\).

---

# Part XXIII — ANOVA / Sum-of-Squares Decomposition

## 48. Definitions

When an intercept is included,

\[
\operatorname{TSS}
=
\sum_{i=1}^n
(y_i-\bar y)^2,
\]

\[
\operatorname{ESS}
=
\sum_{i=1}^n
(\hat y_i-\bar y)^2,
\]

\[
\operatorname{RSS}
=
\operatorname{SSE}
=
\sum_{i=1}^n
(y_i-\hat y_i)^2.
\]

Because

\[
y_i-\bar y
=
(\hat y_i-\bar y)
+
(y_i-\hat y_i),
\]

and the fitted centered component is orthogonal to the residual component,

\[
\boxed{
\operatorname{TSS}
=
\operatorname{ESS}
+
\operatorname{RSS}.
}
\]

---

## 49. Proof

In vector form,

\[
y-\bar y\mathbf 1
=
(\hat y-\bar y\mathbf 1)+e.
\]

With an intercept,

\[
e
\perp
\operatorname{Col}(X).
\]

Since both \(\hat y\) and \(\mathbf 1\) are in \(\operatorname{Col}(X)\),

\[
\hat y-\bar y\mathbf 1
\in
\operatorname{Col}(X).
\]

Hence

\[
e^T(\hat y-\bar y\mathbf 1)=0.
\]

Apply Pythagoras:

\[
\|y-\bar y\mathbf 1\|^2
=
\|\hat y-\bar y\mathbf 1\|^2
+
\|e\|^2.
\]

Therefore

\[
\boxed{
\operatorname{TSS}
=
\operatorname{ESS}
+
\operatorname{RSS}.
}
\]

---

# Part XXIV — \(R^2\)

## 50. Definition

\[
\boxed{
R^2
=
1-
\frac{\operatorname{RSS}}{\operatorname{TSS}}
}
\]

and, using the decomposition,

\[
R^2
=
\frac{\operatorname{ESS}}{\operatorname{TSS}}.
\]

Interpretation:

\[
R^2
\]

is the fraction of centered response variation explained by the fitted model.

### With an intercept

Because OLS cannot fit worse than the intercept-only model,

\[
0\le R^2\le1.
\]

### Without an intercept

The usual centered decomposition may fail and the conventional \(R^2\) can even be negative.

---

# Part XXV — Statistical Inference

## 51. Standard Error of a Coefficient

Estimate \(\sigma^2\) by

\[
s^2
=
\frac{\operatorname{RSS}}{n-p}.
\]

Then

\[
\widehat{\operatorname{Var}}(\hat\beta)
=
s^2(X^TX)^{-1}.
\]

The standard error of coefficient \(j\) is

\[
\boxed{
SE(\hat\beta_j)
=
s\sqrt{[(X^TX)^{-1}]_{jj}}.
}
\]

---

## 52. \(t\)-Test

To test

\[
H_0:\beta_j=\beta_{j,0},
\]

use

\[
\boxed{
t
=
\frac{
\hat\beta_j-\beta_{j,0}
}{
SE(\hat\beta_j)
}.
}
\]

Under Gaussian errors and the classical model,

\[
t\sim t_{n-p}
\]

under \(H_0\).

---

## 53. Confidence Interval

A \(100(1-\alpha)\%\) confidence interval is

\[
\boxed{
\hat\beta_j
\pm
t_{1-\alpha/2,n-p}
SE(\hat\beta_j).
}
\]

---

## 54. Joint \(F\)-Tests

Suppose we test

\[
H_0:R\beta=r
\]

with \(q\) restrictions.

A common form is

\[
F
=
\frac{
(RSS_R-RSS_U)/q
}{
RSS_U/(n-p)
},
\]

where

- \(RSS_R\) = residual sum of squares under restricted model,
- \(RSS_U\) = residual sum of squares under unrestricted model.

Under the classical Gaussian model,

\[
F\sim F_{q,n-p}
\]

under \(H_0\).

---

# Part XXVI — Prediction

## 55. Mean Response at a New Point

For a new feature vector \(x_0\),

\[
\hat y_0
=
x_0^T\hat\beta.
\]

Conditional variance of the estimated mean is

\[
\operatorname{Var}
(
x_0^T\hat\beta
\mid X
)
=
\sigma^2
x_0^T(X^TX)^{-1}x_0.
\]

---

## 56. Prediction of a New Observation

A future observation is

\[
y_0
=
x_0^T\beta+\varepsilon_0.
\]

Prediction error is

\[
y_0-x_0^T\hat\beta.
\]

Its variance is

\[
\boxed{
\sigma^2
\left[
1+
x_0^T(X^TX)^{-1}x_0
\right].
}
\]

The extra \(1\) appears because a future observation contains new irreducible noise.

Hence a prediction interval is wider than a confidence interval for the mean response.

---

# Part XXVII — Leverage

## 57. Diagonal Elements of the Hat Matrix

Let

\[
h_{ii}
\]

be the \(i\)-th diagonal entry of \(H\).

These are called leverage values.

Because

\[
\hat y=Hy,
\]

the fitted value \(\hat y_i\) depends strongly on \(y_i\) when \(h_{ii}\) is large.

Important identities:

\[
0\le h_{ii}\le1,
\]

and if rank is \(p\),

\[
\sum_{i=1}^n h_{ii}
=
\operatorname{tr}(H)
=
p.
\]

Therefore average leverage is

\[
\frac pn.
\]

---

# Part XXVIII — Residual Variance

## 58. Residual Covariance

Since

\[
e=M\varepsilon,
\]

\[
\operatorname{Var}(e\mid X)
=
M(\sigma^2I)M^T.
\]

Because \(M\) is symmetric and idempotent,

\[
\boxed{
\operatorname{Var}(e\mid X)
=
\sigma^2M
=
\sigma^2(I-H).
}
\]

For individual residual \(e_i\),

\[
\boxed{
\operatorname{Var}(e_i\mid X)
=
\sigma^2(1-h_{ii}).
}
\]

Residuals therefore do not all have the same variance, even when the original model errors are homoskedastic.

---

# Part XXIX — OLS Under an Intercept: Extra Identities

With an intercept:

\[
\sum_i e_i=0,
\]

\[
\bar{\hat y}=\bar y,
\]

\[
e^T\hat y=0,
\]

\[
e^T\mathbf 1=0.
\]

Also,

\[
\sum_i x_{ij}e_i=0
\]

for every included predictor column.

These are **sample identities produced by the OLS solution**. They are not assumptions about the population errors.

---

# Part XXX — Error vs Residual

## 59. Error

The model error is

\[
\varepsilon_i
=
y_i-E[y_i\mid x_i].
\]

It depends on the unknown true regression function and is generally unobservable.

## 60. Residual

The residual is

\[
e_i=y_i-\hat y_i.
\]

It is computable after fitting the model.

Do not use the words error and residual as if they were exactly identical.

---

# Part XXXI — What OLS Assumptions Are NOT Needed for Geometry

## 61. Pure Optimization Result

The solution to

\[
\min_\beta
\|y-X\beta\|^2
\]

does not require assumptions such as:

- normal errors,
- independent errors,
- homoskedastic errors,
- zero conditional mean.

Those are statistical assumptions used to interpret or infer from the fitted coefficients.

The projection identity

\[
X^Te=0
\]

is an algebraic consequence of least-squares minimization.

---

# Part XXXII — Multicollinearity

## 62. Perfect Multicollinearity

If

\[
x_3=2x_1-5x_2,
\]

then columns are linearly dependent.

Hence

\[
\operatorname{rank}(X)<p.
\]

Then

\[
X^TX
\]

is singular and the usual inverse formula fails.

## 63. Near Multicollinearity

If one predictor is nearly a linear combination of others, then \(X^TX\) is ill-conditioned.

Consequences include:

- large coefficient variances,
- unstable coefficients,
- sensitivity to small perturbations,
- coefficients with unexpected signs,
- individually insignificant \(t\)-tests even when overall fit is strong.

Predictions may still be reasonably stable in the region of observed data.

---

# Part XXXIII — Ridge Regression Connection

## 64. Objective

Ridge regression minimizes

\[
\|y-X\beta\|^2
+
\lambda\|\beta\|^2.
\]

Differentiate:

\[
-2X^Ty
+
2X^TX\beta
+
2\lambda\beta
=
0.
\]

Therefore

\[
(X^TX+\lambda I)\hat\beta_{\text{ridge}}
=
X^Ty.
\]

So

\[
\boxed{
\hat\beta_{\text{ridge}}
=
(X^TX+\lambda I)^{-1}X^Ty.
}
\]

Ridge can produce a unique solution even when \(X^TX\) is singular, provided the penalized coordinates are handled appropriately.

However, ridge is **not the only way** to resolve nonuniqueness. The Moore–Penrose pseudoinverse also gives a unique minimum-norm OLS solution.

---

# Part XXXIV — Why OLS Is Sensitive to Outliers

## 65. Squared Loss

OLS minimizes

\[
\sum_i e_i^2.
\]

If one residual doubles,

\[
e_i^2
\]

quadruples.

Therefore large residuals receive disproportionately large weight.

This makes OLS sensitive to response outliers.

High-leverage predictor points can be especially influential because they can strongly alter the fitted hyperplane.

---

# Part XXXV — Loss Geometry

## 66. Hessian of the OLS Objective

Recall

\[
S(\beta)
=
y^Ty
-
2\beta^TX^Ty
+
\beta^TX^TX\beta.
\]

Gradient:

\[
\nabla S
=
-2X^Ty
+
2X^TX\beta.
\]

Hessian:

\[
\boxed{
\nabla^2 S
=
2X^TX.
}
\]

Since

\[
X^TX
\]

is always positive semidefinite, \(S(\beta)\) is convex.

If \(X\) has full column rank, \(X^TX\) is positive definite, so \(S(\beta)\) is strictly convex and the minimizer is unique.

If \(X\) is rank deficient, the objective is convex but not strictly convex, so multiple coefficient vectors can minimize it.

---

# Part XXXVI — SVD Interpretation

## 67. Singular Value Decomposition

Let

\[
X=U\Sigma V^T.
\]

If rank is \(r\),

\[
X^+
=
V\Sigma^+U^T.
\]

Then

\[
\hat\beta
=
X^+y
=
V\Sigma^+U^Ty.
\]

The fitted vector is

\[
\hat y
=
XX^+y.
\]

For the thin SVD using only nonzero singular values,

\[
X=U_r\Sigma_rV_r^T,
\]

we get

\[
XX^+
=
U_rU_r^T.
\]

Therefore

\[
\boxed{
\hat y
=
U_rU_r^Ty.
}
\]

This makes the projection interpretation especially transparent: \(U_r\) is an orthonormal basis of \(\operatorname{Col}(X)\), and \(U_rU_r^T\) is the orthogonal projector onto it.

---

# Part XXXVII — QR Interpretation

## 68. QR Decomposition

If \(X\) has full column rank,

\[
X=QR
\]

where

\[
Q^TQ=I
\]

and \(R\) is upper triangular.

Then the least-squares problem becomes

\[
\min_\beta
\|y-QR\beta\|^2.
\]

The normal equations imply

\[
R\hat\beta
=
Q^Ty.
\]

So

\[
\boxed{
\hat\beta=R^{-1}Q^Ty.
}
\]

Numerically, QR is often preferable to explicitly forming

\[
(X^TX)^{-1}.
\]

---

# Part XXXVIII — Numerical Warning

## 69. Do Not Literally Compute the Matrix Inverse When Coding

The formula

\[
\hat\beta=(X^TX)^{-1}X^Ty
\]

is excellent for theory.

In numerical computation, one typically solves the linear system directly or uses QR/SVD.

Why?

Forming \(X^TX\) squares the condition number:

\[
\kappa(X^TX)
\approx
\kappa(X)^2.
\]

Thus numerical instability can become much worse.

---

# Part XXXIX — Bias Under Model Misspecification

## 70. Omitted Variable Bias Idea

Suppose the true model is

\[
y=X\beta+Z\gamma+\varepsilon,
\]

but we regress \(y\) only on \(X\).

Then

\[
\hat\beta
=
(X^TX)^{-1}X^Ty.
\]

Substitute the true model:

\[
\hat\beta
=
\beta
+
(X^TX)^{-1}X^TZ\gamma
+
(X^TX)^{-1}X^T\varepsilon.
\]

If \(Z\) is correlated with \(X\), the middle term generally does not vanish.

This is the matrix form of omitted-variable bias.

---

# Part XL — Centering and the Intercept

## 71. Centered Simple Regression

With an intercept,

\[
b_1
=
\frac{
\sum (x_i-\bar x)(y_i-\bar y)
}{
\sum (x_i-\bar x)^2
}.
\]

If both \(x\) and \(y\) are centered first,

\[
x_i^c=x_i-\bar x,
\]

\[
y_i^c=y_i-\bar y,
\]

then the intercept in centered coordinates is zero.

Centering clarifies why the slope depends only on deviations around the means.

---

# Part XLI — Covariance and Correlation Form

## 72. Slope in Simple Regression

Using sample covariance and variance with the same denominator convention,

\[
b_1
=
\frac{\operatorname{Cov}(x,y)}
{\operatorname{Var}(x)}.
\]

If

\[
r_{xy}
=
\frac{\operatorname{Cov}(x,y)}
{s_xs_y},
\]

then

\[
\boxed{
b_1
=
r_{xy}
\frac{s_y}{s_x}.
}
\]

This shows:

- sign of slope = sign of correlation,
- slope also depends on units,
- correlation is unitless but slope is not.

---

# Part XLII — \(R^2\) in Simple Regression

## 73. Relation to Correlation

In simple linear regression with an intercept,

\[
\boxed{
R^2=r_{xy}^2.
}
\]

This special identity does not generalize to arbitrary multiple regression in the same simple way.

---

# Part XLIII — OLS as a Projection Operator

## 74. Projection Characterization

The fitted vector \(\hat y\) satisfies both:

\[
\hat y\in\operatorname{Col}(X),
\]

and

\[
y-\hat y
\perp
\operatorname{Col}(X).
\]

These two conditions uniquely characterize the orthogonal projection.

This is the deepest geometric statement behind ordinary least squares.

---

# Part XLIV — Exact Projection Proof Using Normal Equations

Suppose \(\hat\beta\) solves

\[
X^TX\hat\beta=X^Ty.
\]

Then

\[
X^T(y-X\hat\beta)=0.
\]

Thus

\[
e=y-X\hat\beta
\]

is in

\[
\operatorname{Null}(X^T).
\]

But a standard linear algebra result is

\[
\operatorname{Null}(X^T)
=
\operatorname{Col}(X)^\perp.
\]

Therefore

\[
e
\perp
\operatorname{Col}(X).
\]

Since

\[
\hat y=X\hat\beta\in\operatorname{Col}(X),
\]

\(\hat y\) is exactly the orthogonal projection of \(y\) onto \(\operatorname{Col}(X)\).

---

# Part XLV — Proof of
\[
\operatorname{Null}(X^T)=\operatorname{Col}(X)^\perp
\]

Let \(z\in\operatorname{Null}(X^T)\).

Then

\[
X^Tz=0.
\]

This means

\[
x_j^Tz=0
\]

for every column \(x_j\) of \(X\).

Thus \(z\) is perpendicular to every column of \(X\).

Any vector in \(\operatorname{Col}(X)\) has the form

\[
v=\sum_j c_jx_j.
\]

Then

\[
z^Tv
=
z^T
\sum_j c_jx_j
=
\sum_j c_jz^Tx_j
=
0.
\]

Therefore

\[
z\perp\operatorname{Col}(X).
\]

Hence

\[
\operatorname{Null}(X^T)
\subseteq
\operatorname{Col}(X)^\perp.
\]

The dimensions of the two spaces are equal:

\[
\dim\operatorname{Null}(X^T)
=
n-\operatorname{rank}(X),
\]

\[
\dim\operatorname{Col}(X)^\perp
=
n-\operatorname{rank}(X).
\]

Therefore the spaces are equal.

---

# Part XLVI — Why the Normal Equations Are Called “Normal”

In geometry, a vector normal to a subspace is perpendicular to it.

The normal equations force

\[
e=y-X\hat\beta
\]

to be normal to the column space:

\[
X^Te=0.
\]

Hence the name **normal equations**.

---

# Part XLVII — PDF Proof Check and Corrections

This section directly checks the main claims in the uploaded PDF.

## 75. Correct: OLS residual is orthogonal in column space

The PDF states that in \(\mathbb R^n\),

\[
X^Te=0.
\]

This is correct.

The important interpretation is:

- columns of \(X\) are vectors in sample space,
- \(\hat y\in\operatorname{Col}(X)\),
- \(e\in\operatorname{Col}(X)^\perp\).

---

## 76. Correct: OLS uses vertical residuals in an ordinary scatterplot

The PDF distinguishes vertical OLS residuals from perpendicular distances in the ordinary \(x\)-\(y\) picture.

This is correct.

OLS solves

\[
\min
\sum_i
(y_i-\hat y_i)^2,
\]

not

\[
\min
\sum_i
d_{\perp,i}^2.
\]

---

## 77. Correct with qualification: “one dimension per sample”

It is valid to represent a response vector with \(n\) coordinates in \(\mathbb R^n\).

However, statistical independence of observations is **not necessary** for this vector-space representation.

The standard coordinate axes are orthogonal by construction.

---

## 78. Correction: \(p=n\) does not automatically span \(\mathbb R^n\)

The statement is true only if

\[
\operatorname{rank}(X)=n.
\]

A square matrix can still be singular.

---

## 79. Correction: \(n<p\) does not always imply zero training error

Zero training error for every \(y\) requires

\[
\operatorname{Col}(X)=\mathbb R^n,
\]

which is equivalent to

\[
\operatorname{rank}(X)=n.
\]

If rank is less than \(n\), some response vectors remain unreachable.

---

## 80. Correction / qualification: ridge is not required merely because \(X^TX\) is singular

Possible approaches include:

- Moore–Penrose pseudoinverse,
- QR with pivoting,
- SVD,
- constraints,
- ridge or other regularization.

Ridge is useful, but not mathematically required just to define one least-squares solution.

---

## 81. Qualification: TLS and units

Ordinary Euclidean TLS is sensitive to relative axis scaling.

If variables have different units, raw Euclidean geometry may be inappropriate.

But generalized TLS can account for unequal measurement-error scales or covariance structures.

---

# Part XLVIII — Common Exam Questions and Short Answers

## 82. Why is \(X^Te=0\)?

Because the OLS residual is orthogonal to every column of \(X\), i.e. to the entire model subspace.

---

## 83. Why does \(\hat y^Te=0\)?

Because

\[
\hat y=X\hat\beta
\]

is a linear combination of columns of \(X\), and \(e\) is orthogonal to each of them.

---

## 84. Why do residuals sum to zero?

If an intercept is included, the all-ones vector is a column of \(X\). Since

\[
X^Te=0,
\]

we get

\[
\mathbf1^Te=\sum_i e_i=0.
\]

---

## 85. When is \((X^TX)^{-1}\) valid?

When

\[
\operatorname{rank}(X)=p.
\]

That is, the columns of \(X\) must be linearly independent.

---

## 86. What does OLS minimize?

\[
\boxed{
\operatorname{SSE}
=
\sum_i
(y_i-\hat y_i)^2
=
\|y-X\beta\|^2.
}
\]

---

## 87. Is OLS the same as perpendicular-distance regression in the \(x\)-\(y\) plot?

No.

OLS minimizes vertical response errors.

Orthogonal regression / TLS minimizes perpendicular geometric distances in the plotted predictor-response plane.

---

## 88. Does OLS require Gaussian errors?

No.

Gaussian errors are not required for the coefficient minimizer or unbiasedness.

They are used for exact likelihood and finite-sample \(t/F\) inference.

---

## 89. What is the most important assumption for unbiasedness?

\[
\boxed{
E[\varepsilon\mid X]=0.
}
\]

---

## 90. What is the geometric meaning of \(H\)?

\[
H
=
X(X^TX)^{-1}X^T
\]

is the orthogonal projector from \(\mathbb R^n\) onto \(\operatorname{Col}(X)\).

---

## 91. What is the geometric meaning of \(I-H\)?

\[
I-H
\]

projects onto

\[
\operatorname{Col}(X)^\perp
=
\operatorname{Null}(X^T).
\]

---

# Part XLIX — Compact Formula Sheet

## Model

\[
y=X\beta+\varepsilon
\]

## OLS objective

\[
\hat\beta
=
\arg\min_\beta
\|y-X\beta\|^2
\]

## Normal equations

\[
X^TX\hat\beta=X^Ty
\]

## Closed form

\[
\hat\beta
=
(X^TX)^{-1}X^Ty
\]

when \(X\) has full column rank.

## Fitted values

\[
\hat y=X\hat\beta=Hy
\]

## Hat matrix

\[
H=X(X^TX)^{-1}X^T
\]

## Residuals

\[
e=y-\hat y
\]

## Residual maker

\[
M=I-H
\]

## Orthogonality

\[
X^Te=0
\]

\[
\hat y^Te=0
\]

## Unbiasedness

\[
E[\hat\beta\mid X]=\beta
\]

if

\[
E[\varepsilon\mid X]=0.
\]

## Variance

\[
\operatorname{Var}(\hat\beta\mid X)
=
\sigma^2(X^TX)^{-1}
\]

under

\[
\operatorname{Var}(\varepsilon\mid X)
=
\sigma^2I.
\]

## Error variance estimator

\[
s^2
=
\frac{\operatorname{RSS}}{n-p}.
\]

## Sum-of-squares decomposition

\[
TSS=ESS+RSS
\]

with an intercept.

## \(R^2\)

\[
R^2
=
1-\frac{RSS}{TSS}.
\]

## Simple-regression slope

\[
\hat\beta_1
=
\frac{
\sum
(x_i-\bar x)(y_i-\bar y)
}{
\sum
(x_i-\bar x)^2
}.
\]

## Simple-regression intercept

\[
\hat\beta_0
=
\bar y-\hat\beta_1\bar x.
\]

---

# Part L — Concept Map

A useful way to remember linear regression is:

\[
\boxed{
\text{Data}
\to
\text{Matrix model}
\to
\text{Least-squares minimization}
\to
\text{Normal equations}
\to
\text{Orthogonal projection}
}
\]

then, after adding probability assumptions,

\[
\boxed{
\text{Exogeneity}
\to
\text{Unbiasedness}
}
\]

\[
\boxed{
\text{Homoskedastic uncorrelated errors}
\to
\text{Gauss–Markov / BLUE}
}
\]

\[
\boxed{
\text{Gaussian errors}
\to
\text{MLE + exact }t/F\text{ inference}
}
\]

The central mathematical identity is

\[
\boxed{
X^T(y-X\hat\beta)=0.
}
\]

From this one condition come:

- normal equations,
- residual-feature orthogonality,
- residual-fitted-value orthogonality,
- zero residual sum when an intercept exists,
- projection geometry,
- Pythagorean sum-of-squares decompositions.

---

# Source-Check Notes

The uploaded PDF provided the core framing for:

- regression vs classification,
- OLS residual geometry,
- data-space vs column-space distinction,
- \(\operatorname{Col}(X)\),
- \(X^Te=0\),
- normal equations,
- matrix-calculus derivation,
- projection / hat matrix,
- \(Ax=b\) interpretation,
- dimension / rank discussion.

The extended material in these notes—Gauss–Markov, unbiasedness, variance, Gaussian MLE, SVD/QR, residual variance, inference, and the detailed qualifications—is added to make the notes mathematically complete and to correct or refine claims that were too broad in the source.

