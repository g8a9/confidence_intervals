Confidence Intervals
====================

<!-- [![image](https://img.shields.io/pypi/v/confidence_intervals.svg)](https://pypi.python.org/pypi/confidence_intervals) -->

<!-- [![image](https://img.shields.io/travis/g8a9/confidence_intervals.svg)](https://travis-ci.com/g8a9/confidence_intervals) -->

<!-- [![Documentation Status](https://readthedocs.org/projects/confidence-intervals/badge/?version=latest)](https://confidence-intervals.readthedocs.io/en/latest/?version=latest) -->

Simple evaluation of classification confidence intervals.

```bash
pip install git+https://github.com/g8a9/confidence_intervals.git
```

<!-- -   Free software: MIT license -->
<!-- -   Documentation: <https://confidence-intervals.readthedocs.io>. -->

### Getting Started

```python
clf = RandomForestClassifier().fit(X_train, y_train)
y_pred = clf.predict(X_test)
bootstrap_test(y_test, y_pred)

# Output
>>> ConfidenceResult(n_iter=200, ci_level=95, mean=0.9777111111111112, ci_lower=0.9643888888888889, ci_upper=0.9888888888888889)
```

## Features

Confidence Intervals Estimation by means of:

- bootstrap sampling on test set predictions (use `n_iter > 200`)
- simple estimation over multiple runs (useful for CI estimation across multiple random initializations)

<!-- Features
--------

-   TODO -->

Credits
-------

This package was created with
[Cookiecutter](https://github.com/audreyr/cookiecutter) and the
[audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage)
project template.
