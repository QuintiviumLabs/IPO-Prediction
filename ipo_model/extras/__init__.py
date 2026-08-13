"""Optional add-on model rungs.

Each module here is self-contained and imports its third-party dependency
lazily, so the core pipeline runs unchanged whether or not these are
installed. They reuse `baselines.common.run_folds`, which means they see
exactly the same purged walk-forward folds, per-fold scalers, engineered
feature table and metrics as every other rung — results are directly
comparable.

    pip install -r requirements-extras.txt

  linear        Ridge / Lasso / ElasticNet (scikit-learn)
  tabpfn_model  TabPFN in-context learning (tabpfn)
"""
