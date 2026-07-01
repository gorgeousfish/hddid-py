"""Sphinx configuration for hddid documentation."""
import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

project = "hddid"
copyright = "2020, Yang Ning, Sida Peng, Jing Tao"
author = "Yang Ning, Sida Peng, Jing Tao"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx_autodoc_typehints",
    "myst_parser",
]

# Napoleon settings for NumPy-style docstrings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True

# MyST for Markdown support
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "api.md", "api-reference.md", "release-checklist.md", "release-diagnosis.md", "validation-lanes.md"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

# Allow an offline build to skip remote inventory fetching.  In offline mode
# external cross-references are not resolved, but the build completes without
# SSL warnings.
if os.environ.get("HDDID_DOCS_OFFLINE"):
    intersphinx_mapping = {}

autodoc_member_order = "bysource"
autodoc_typehints = "description"
suppress_warnings = ["intersphinx.external"]
