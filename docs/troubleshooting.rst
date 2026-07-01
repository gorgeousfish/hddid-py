Troubleshooting
===============

Common Errors
-------------

ZeroValidHoldoutError
~~~~~~~~~~~~~~~~~~~~~

**Symptom**: ``ZeroValidHoldoutError: fold X has zero valid holdout observations``

**Cause**: All observations in a cross-fitting fold have propensity scores
outside the trimming bounds ``[trim_lower, trim_upper]``.

**Fix**: Reduce ``p`` (number of covariates), increase ``n`` (sample size),
or widen the trimming bounds::

   fit_hddid(..., trim_lower=0.005, trim_upper=0.995)

SparseDirectionInfeasibleError
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptom**: Linear program for Eq. (4.2) Dantzig selector is infeasible.

**Cause**: The covariance matrix is ill-conditioned or ``lambda_prime`` is too tight.

**Fix**: Increase ``lambda_prime`` or check that ``x`` has sufficient variation.

Inference exceptions
~~~~~~~~~~~~~~~~~~~~

The inference layer raises typed exceptions that all inherit from
``InferenceComputationError`` (a subclass of ``RuntimeError``):

- ``InvalidInferenceInputError`` -- the score payload and estimation payload are
  inconsistent (e.g., mismatched sample sizes or basis dimensions), or a
  required grid is empty.
- ``MissingEvaluationGridError`` -- a nonparametric inference call was made
  without providing evaluation points in ``z0``.
- ``SingularCovarianceError`` -- a covariance matrix that must be inverted is
  numerically singular. Often caused by a rank-deficient basis or nearly
  collinear covariates.
- ``NonpositiveVarianceError`` -- the estimated pointwise or uniform-band
  variance is not strictly positive. Check for near-constant outcomes or an
  extremely small effective sample size after trimming.
- ``SparseDirectionInfeasibleError`` -- the linear program for Eq. (4.2) or
  Eq. (4.3) has no feasible solution. Relax the tuning parameter or increase
  the sample size.
- ``InferenceComputationError`` -- base class for the errors above; also raised
  directly for unexpected numerical failures during bootstrap or matrix-root
  computations.

Estimation exceptions
~~~~~~~~~~~~~~~~~~~~~

- ``Eq31SolverConvergenceError`` -- the coordinate-descent solver for Eq. (3.1)
  did not converge within ``max_iter`` iterations. Try increasing ``max_iter``,
  loosening ``tol``, or using ``solver="sklearn"``.
- ``Eq31ProjectionRankError`` -- the sieve basis matrix is rank-deficient.
  Reduce the basis degree or switch to a different ``basis_family``.

Installation Issues
-------------------

Missing optional dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``solver="sklearn"`` requires scikit-learn: ``pip install scikit-learn``
- ``HDDIDFit.from_dataframe()`` requires pandas: ``pip install pandas``
- Documentation build requires: ``pip install hddid[docs]``

Performance Tips
----------------

- Use ``n_jobs=4`` for parallel cross-fitting on multi-core machines
- Use ``solver="sklearn"`` for faster LASSO convergence in high dimensions
- Use ``basis_family="bspline"`` for optimal approximation properties
- Use ``penalty_lambda="auto"`` for theory-guided regularization

Regularization Parameter Selection
-----------------------------------

The ``penalty_lambda`` argument of :func:`hddid.fit_hddid` controls the L1
penalty on the parametric coefficients in Eq. (3.1):

- ``penalty_lambda=0.0`` (default): unpenalized least-squares projection. Use
  this when ``p < n`` or when you do not need variable selection.
- ``penalty_lambda="auto"``: data-driven choice
  :math:`\lambda = 2.2\sqrt{\log(p)/n_{\text{valid}}}`, where
  :math:`n_{\text{valid}}` is the trimmed effective sample size. This matches
  the rate used in the theoretical analysis and is a good starting point for
  high-dimensional settings.
- A positive float: a user-supplied penalty. Larger values produce sparser
  coefficient vectors; values that are too large can remove all signal.

When ``p`` is large relative to ``n``, start with ``"auto"`` and inspect the
parametric estimates. If many coefficients are shrunk to exactly zero while
standard errors remain large, consider a slightly smaller penalty.
