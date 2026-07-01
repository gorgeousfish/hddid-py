Mathematical Background
========================

The ``hddid`` package implements the doubly robust semiparametric
difference-in-differences estimator from:

   Ning, Y., Peng, S., & Tao, J. (2020). Doubly robust semiparametric
   difference-in-differences estimators with high-dimensional data.
   arXiv preprint arXiv:2009.03151.

Model
-----

The partially linear model is:

.. math::

   E[\Delta Y_i | X_i, Z_i] = X_i'\beta_0 + f_0(Z_i)

where :math:`\beta_0 \in \mathbb{R}^p` (sparse, :math:`p \gg n`) and
:math:`f_0(\cdot)` is a smooth nonparametric function.

Key Equations
-------------

**Eq. (2.5)/(2.7): Doubly Robust Score**

Implemented by :func:`hddid.build_score_payload`.

.. math::

   \psi(W_i; \alpha_0) = \rho_0 \times [\Delta Y_i - (1-\pi)\Phi_1(W_i) - \pi\Phi_0(W_i)]

**Eq. (3.1): Second-Stage LASSO**

Implemented by :func:`hddid.estimate_eq31_mainline` and called automatically by
:func:`hddid.fit_hddid`.

.. math::

   (\hat\beta, \hat\gamma) = \arg\min_{\beta, \gamma}
   \mathbb{E}_n[(\psi_i - X_i'\beta - \Psi_n(Z_i)'\gamma)^2] + \lambda\|\beta\|_1

**Eq. (4.2): Dantzig Selector**

Implemented by :func:`hddid.solve_eq42_sparse_direction` and used inside
:func:`hddid.estimate_parametric_inference`.

.. math::

   \hat{w} = \arg\min_w \|w\|_1 \quad \text{s.t.} \quad
   \|\hat\Sigma_{\tilde X} w + \xi\|_\infty \leq \lambda'

**Eq. (4.3): Nonparametric Projection**

Implemented by :func:`hddid.solve_eq43_projection_matrix` and used inside
:func:`hddid.estimate_nonparametric_inference`.

.. math::

   \hat{M} = \arg\min_M \|M\|_{1,\text{row}} \quad \text{s.t. constraints}

Cross-Fitting and Neyman Orthogonality
--------------------------------------

The first-stage propensity score and conditional outcome means are estimated
with sample splitting so that the nuisance models are trained on data separate
from the observations that contribute to the doubly-robust score.  This
sample-splitting (implemented by :func:`hddid.make_crossfit_splits` and
:class:`hddid.CrossfitNuisanceEstimator`) removes the first-order bias from
plug-in nuisance estimation and makes the score Neyman-orthogonal.  As a
result, the second-stage LASSO in Eq. (3.1) remains valid when the nuisance
estimators converge at slower-than-parametric rates.
