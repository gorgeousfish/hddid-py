Inference
=========

.. autofunction:: hddid.estimate_parametric_inference
.. autofunction:: hddid.estimate_nonparametric_inference
.. autofunction:: hddid.solve_eq42_sparse_direction
.. autofunction:: hddid.solve_eq43_projection_matrix
.. autofunction:: hddid.diagnose_nonparametric_omega_f

Exceptions
----------

.. autoclass:: hddid.InferenceComputationError
.. autoclass:: hddid.InvalidInferenceInputError
.. autoclass:: hddid.MissingEvaluationGridError
.. autoclass:: hddid.SingularCovarianceError
.. autoclass:: hddid.SparseDirectionInfeasibleError
.. autoclass:: hddid.NonpositiveVarianceError

Payload Classes
---------------

.. autoclass:: hddid.ParametricInferencePayload
   :members:
.. autoclass:: hddid.NonparametricInferencePayload
   :members:
