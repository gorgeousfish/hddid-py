"""Verify that the inference module split maintains backward compatibility.

This test suite ensures:
1. All public names are importable from both hddid.inference and hddid
2. No circular imports exist
3. Module structure is correct
"""
import pytest


class TestBackwardCompatibility:
    """API backward compatibility after module split."""

    def test_all_public_names_importable_from_inference_module(self):
        """All original public names must be importable from hddid.inference."""
        from hddid.inference import (
            InferenceComputationError,
            InvalidInferenceInputError,
            MissingEvaluationGridError,
            NonpositiveVarianceError,
            SingularCovarianceError,
            SparseDirectionInfeasibleError,
            ParametricInferencePayload,
            NonparametricInferencePayload,
            solve_eq42_sparse_direction,
            solve_eq43_projection_matrix,
            diagnose_nonparametric_omega_f,
            estimate_parametric_inference,
            estimate_nonparametric_inference,
        )
        # Verify they are callable/class as expected
        assert issubclass(InferenceComputationError, RuntimeError)
        assert issubclass(InvalidInferenceInputError, InferenceComputationError)
        assert issubclass(MissingEvaluationGridError, InferenceComputationError)
        assert issubclass(NonpositiveVarianceError, InferenceComputationError)
        assert issubclass(SingularCovarianceError, InferenceComputationError)
        assert issubclass(SparseDirectionInfeasibleError, InferenceComputationError)
        assert callable(solve_eq42_sparse_direction)
        assert callable(solve_eq43_projection_matrix)
        assert callable(diagnose_nonparametric_omega_f)
        assert callable(estimate_parametric_inference)
        assert callable(estimate_nonparametric_inference)

    def test_all_public_names_importable_from_top_level(self):
        """All public names must also be importable from hddid package."""
        from hddid import (
            InferenceComputationError,
            InvalidInferenceInputError,
            MissingEvaluationGridError,
            NonpositiveVarianceError,
            SingularCovarianceError,
            SparseDirectionInfeasibleError,
            ParametricInferencePayload,
            NonparametricInferencePayload,
            solve_eq42_sparse_direction,
            solve_eq43_projection_matrix,
            diagnose_nonparametric_omega_f,
            estimate_parametric_inference,
            estimate_nonparametric_inference,
        )
        assert callable(estimate_parametric_inference)
        assert callable(estimate_nonparametric_inference)

    def test_no_circular_imports(self):
        """Verify no circular import issues in the split modules."""
        import importlib
        import sys

        # Clear any cached modules
        modules_to_clear = [
            k for k in sys.modules if k.startswith("hddid")
        ]
        for mod in modules_to_clear:
            del sys.modules[mod]

        # Import each submodule independently
        importlib.import_module("hddid._inference_common")
        importlib.import_module("hddid.inference_bootstrap")
        importlib.import_module("hddid.inference_parametric")
        importlib.import_module("hddid.inference_nonparametric")
        importlib.import_module("hddid.inference")
        importlib.import_module("hddid")

    def test_submodule_direct_import(self):
        """Submodules can be imported directly for targeted use."""
        from hddid.inference_parametric import (
            solve_eq42_sparse_direction,
            estimate_parametric_inference,
            ParametricInferencePayload,
        )
        from hddid.inference_nonparametric import (
            solve_eq43_projection_matrix,
            estimate_nonparametric_inference,
            NonparametricInferencePayload,
            diagnose_nonparametric_omega_f,
        )
        from hddid.inference_bootstrap import (
            compute_uniform_band_critical_value,
        )
        assert callable(compute_uniform_band_critical_value)

    def test_exception_hierarchy_preserved(self):
        """Exception class hierarchy must be identical after split."""
        from hddid.inference import (
            InferenceComputationError,
            InvalidInferenceInputError,
            MissingEvaluationGridError,
            SingularCovarianceError,
            SparseDirectionInfeasibleError,
            NonpositiveVarianceError,
        )
        # All subclasses of InferenceComputationError
        assert issubclass(InvalidInferenceInputError, InferenceComputationError)
        assert issubclass(MissingEvaluationGridError, InferenceComputationError)
        assert issubclass(SingularCovarianceError, InferenceComputationError)
        assert issubclass(SparseDirectionInfeasibleError, InferenceComputationError)
        assert issubclass(NonpositiveVarianceError, InferenceComputationError)
        # InferenceComputationError is a RuntimeError
        assert issubclass(InferenceComputationError, RuntimeError)

    def test_inference_module_all_attribute(self):
        """inference.__all__ must contain all expected names."""
        import hddid.inference as inf
        expected = {
            "InferenceComputationError",
            "InvalidInferenceInputError",
            "MissingEvaluationGridError",
            "NonpositiveVarianceError",
            "SingularCovarianceError",
            "SparseDirectionInfeasibleError",
            "ParametricInferencePayload",
            "NonparametricInferencePayload",
            "solve_eq42_sparse_direction",
            "solve_eq43_projection_matrix",
            "diagnose_nonparametric_omega_f",
            "estimate_parametric_inference",
            "estimate_nonparametric_inference",
        }
        assert set(inf.__all__) == expected
