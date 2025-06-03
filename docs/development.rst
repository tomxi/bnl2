Development
===========

Setup
-----

.. code-block:: bash

    git clone https://github.com/tomxi/bnl.git
    cd bnl
    pip install -e ".[dev,docs]"

Documentation
-------------

**Serve Docs**: ``Cmd+Shift+P`` → ``Tasks: Run Task`` → ``Serve Docs``

Builds and serves docs at http://localhost:8000

**Quick Rebuild**: ``cd docs && make html``

Adding New Modules
------------------

1. Create the module in ``src/bnl/``
2. Add imports to ``src/bnl/__init__.py``
3. Update ``docs/api/bnl.rst`` if needed
4. Add examples to ``docs/examples.rst``

Documentation Style
-------------------

Use NumPy-style docstrings:

.. code-block:: python

    def example_function(param1, param2=None):
        """Brief description.

        Parameters
        ----------
        param1 : type
            Description.
        param2 : type, optional
            Description, by default None.

        Returns
        -------
        type
            Description.
        """
        pass 