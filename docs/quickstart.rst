Quick Start
===========

Minimal workflow
----------------

.. code-block:: python

   import numpy as np
   from hddid import fit_hddid

   # Generate example data
   np.random.seed(42)
   n, p = 200, 10
   x = np.random.randn(n, p)
   z = np.random.randn(n)
   treat = (np.random.rand(n) > 0.5).astype(float)
   y0 = x @ np.random.randn(p) * 0.1 + np.random.randn(n)
   y1 = y0 + 0.5 * treat + np.random.randn(n) * 0.1

   # Fit the HDDID model
   fit = fit_hddid(
       y0=y0, y1=y1, treat=treat, x=x, z=z,
       z0=np.array([-1.0, 0.0, 1.0]),
       basis_family="polynomial",
       basis_degree=2,
   )

   # View results
   print(fit.summary(format="text"))

Example output (generated with ``seed=42``, ``n=200``, ``p=10``, polynomial
basis of degree 2):

.. code-block:: text

   --- HDDID Estimation Results ---
   Basis: polynomial (degree=2)
   Folds: 2 | Valid obs: 200 | Alpha: 0.1

   Parametric Estimates:
     Name            Estimate    SE          95% CI
     beta_hat[1]     0.0057      -           -
     beta_hat[2]     0.0103      -           -
     beta_hat[3]     -0.0189     -           -
     beta_hat[4]     0.0032      -           -
     beta_hat[5]     -0.0231     -           -
     beta_hat[6]     -0.0332     -           -
     beta_hat[7]     -0.0212     -           -
     beta_hat[8]     0.0028      -           -
     beta_hat[9]     0.0192      -           -
     beta_hat[10]    0.0120      -           -
   Nonparametric Estimates:
     Name            Estimate    SE          95% CI
     gamma_hat[1]    0.4458      -           -
     gamma_hat[2]    -0.0445     -           -
     gamma_hat[3]    0.0586      -           -
     f(z0=-1.00)     0.5489      -           -
     f(z0=0.00)      0.4458      -           -
     f(z0=1.00)      0.4600      -           -

Choosing a basis family and solver
------------------------------------

The ``basis_family`` argument selects the sieve basis used for the
nonparametric component :math:`f(Z)` in Eq. (3.1):

- ``"polynomial"`` (default): powers :math:`[1, z, z^2, \dots, z^{\text{degree}}]`.
  Simple and fast; a good default for smooth, low-dimensional :math:`Z`.
- ``"trigonometric"``: Fourier basis :math:`[1, \cos(2\pi z), \sin(2\pi z), \dots]`.
  Useful when :math:`f(\cdot)` is periodic.
- ``"bspline"``: cubic B-splines with quantile-based interior knots. Often gives
  the best approximation properties because B-splines have the optimal Bernstein
  constant; see :func:`hddid.bspline_sieve_basis` and
  :func:`hddid.suggest_basis_degree`.

The ``solver`` argument selects the Lasso backend for the parametric block in
Eq. (3.1):

- ``"sklearn"`` (default): scikit-learn's Lasso. Fast and supports warm starts,
  but requires scikit-learn to be installed.
- ``"builtin"`` / ``"native"``: a pure-NumPy coordinate-descent implementation.
  Has no extra dependencies and is bit-for-bit reproducible across platforms.

Using B-spline basis (optimal Bernstein constant)
-------------------------------------------------

.. code-block:: python

   fit_bspline = fit_hddid(
       y0=y0, y1=y1, treat=treat, x=x, z=z,
       z0=np.array([0.0]),
       basis_family="bspline",
       basis_degree=4,
   )

Auto penalty selection
-----------------------

.. code-block:: python

   fit_auto = fit_hddid(
       y0=y0, y1=y1, treat=treat, x=x, z=z,
       z0=np.array([0.0]),
       penalty_lambda="auto",  # λ = 2.2√(log(p)/n_valid), n_valid = trimmed sample size
   )

Advanced: custom nuisance estimators
-------------------------------------

By default, ``fit_hddid`` uses an IRLS logistic regression for the propensity
score and OLS for the conditional outcome means.  You can replace the default
nuisance models with any sklearn-compatible estimator by passing
``nuisance_estimator``.  Pass a dict with ``"propensity"`` and ``"outcome"``
keys to customize each stage separately:

.. code-block:: python

   from sklearn.linear_model import LogisticRegression, Ridge

   fit_sklearn = fit_hddid(
       y0=y0, y1=y1, treat=treat, x=x, z=z,
       z0=np.array([0.0]),
       nuisance_estimator={
           "propensity": LogisticRegression(max_iter=1000),
           "outcome": Ridge(),
       },
   )

The propensity estimator must implement ``.fit(X, y)`` and
``.predict_proba(X)``; the outcome estimator must implement ``.fit(X, y)`` and
``.predict(X)``.  For details see :func:`hddid.fit_hddid`.
