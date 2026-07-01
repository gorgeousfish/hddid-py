Installation
============

From PyPI (when available):

.. code-block:: bash

   pip install hddid

From source:

.. code-block:: bash

   pip install git+https://github.com/gorgeousfish/hddid-py.git

Dependencies
------------

Required:

- Python >= 3.10
- numpy >= 1.24
- scipy >= 1.10
- PyYAML >= 6.0 (used by internal automation-state tools)

Optional:

- scikit-learn (for ``solver="sklearn"`` backend)
- pandas (for ``HDDIDFit.from_dataframe()``)
