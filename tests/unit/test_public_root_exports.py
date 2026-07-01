def test_root_exports_documented_workflow_building_blocks():
    import hddid

    expected = {
        "CrossfitFold",
        "CrossfitPlan",
        "make_crossfit_splits",
        "CrossfitNuisanceEstimator",
        "NuisancePayload",
        "HDDIDFit",
        "fit_hddid",
        "__version__",
    }

    assert expected.issubset(set(hddid.__all__))
    for name in expected:
        assert getattr(hddid, name) is not None
    assert hddid.__version__ == "0.1.0"
