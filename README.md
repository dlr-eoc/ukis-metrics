# [![UKIS](img/ukis-logo.png)](https://www.dlr.de/eoc/en/desktopdefault.aspx/tabid-5413/10560_read-21914/) ukis-metrics

![ukis-metrics](https://github.com/dlr-eoc/ukis-metrics/workflows/ukis-metrics/badge.svg)
[![codecov](https://codecov.io/gh/dlr-eoc/ukis-metrics/branch/main/graph/badge.svg)](https://codecov.io/gh/dlr-eoc/ukis-metrics)
![Upload Python Package](https://github.com/dlr-eoc/ukis-metrics/workflows/Upload%20Python%20Package/badge.svg)
[![PyPI version](https://img.shields.io/pypi/v/ukis-metrics)](https://pypi.python.org/pypi/ukis-metrics/)
[![GitHub license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://black.readthedocs.io/en/stable/)

A pure Numpy-based implementation of the most common performance metrics for semantic image segmentation. 

## Installation
```shell
pip install ukis_metrics
```

```shell
>>> import ukis_metrics
>>> ukis_metrics.__version__
'0.1.3'
```

## Why?
Simply because we wanted a lightweight and fast alternative to scikit learn for tracking during training. 

[execution_time.py](https://github.com/dlr-eoc/ukis-metrics/blob/main/performance/execution_time.py)
compares the execution time of ukis-metrics with sklearn. Here's an example output:
```
Shape of array: (256, 256, 2)

                                        ### Metrics execution time [s] ###

                ukis-metrics                            sklearn metrics         speed gain
acc             0.001900                                0.007627                4.01
rec             0.001716                                0.024509                14.28
pre             0.001815                                0.025021                13.79
f1              0.001798                                0.024770                13.78
iou             0.001797                                0.024247                13.49
kap             0.001824                                0.034577                18.96
```

## Workings and included metrics
In a first step the **true positives** *tp*, **true negatives** *tn*, **false positives** *fp*, **false negatives** *fn*
and the number of valid pixels **n_valid_pixels** are computed. These values are then used to compute the following 
metrics:
- Accuracy [1]:   
  ```math
  acc = \frac{tp + tn}{tp + fn + fp + tn}
  ```
- Recall [1]:
  ```math
  rec = \frac{tp}{tp + fn}
  ```
- Precision [1]:
  ```math
  prec = \frac{tp}{tp + fp}
  ```
- F1-score [2]:
  ```math
  F1 = \frac{2 * prec * rec}{prec + rec}
  ```
- IoU [3]:
  ```math
  IoU = \frac{tp}{tp + fp + fn}
  ```  
- Kappa: The computation of the Kappa-score incorporates several steps. Please refer to [4] for the full 
  documentation

## How to use
Simply pass a Numpy ndarray to get a dict containing the `tpfptnfn` and pass the dict to `segmentation_metrics(tpfptnfn)`:
```python 
import ukis_metrics.seg_metrics as segm
import numpy as np
# ndarray containing the reference data, e.g.
shape = (256, 256, 1)
y_true = np.ones(shape)
# ndarray containing the model predicions, e.g.
y_pred = np.ones(shape)
# get tp, fp, tn, fn an n_valid_pixel
tpfptnfn = segm.tpfptnfn(y_true, y_pred, None)
metrics = segm.segmentation_metrics(tpfptnfn)
```
So far these metrics were only used for binary classification, although one should be able to use them for 
multiclass segmentation too, if the slices for a given class are provided individually.

There is a [jupyter notebook](https://github.com/dlr-eoc/ukis-metrics/tree/main/examples) showing how ukis-metrics can be used and how it could be extended for multiclass problems.

## References
- [1] [Sokolova and Lapalme 2009: A systematic analysis of performance measures for classification tasks](https://www.researchgate.net/publication/222674734_A_systematic_analysis_of_performance_measures_for_classification_tasks)
- [2] [scikit-learn](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [3] [Berman et al. 2018: The Lovász-Softmax loss: A tractable surrogate for the optimization of 
  the intersection-over-union measure in neural networks](https://arxiv.org/pdf/1705.08790.pdf)
- [4] [Tang et al. 2015: Kappa coefficient: a popular measure of rater agreement](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4372765/) 


## Contributors
The UKIS team creates and adapts libraries which simplify the usage of satellite data. Our team includes (in alphabetical order):
* Fichtner, Florian
* Helleis, Max
* Martinis, Sandro
* Wieland, Marc

German Aerospace Center (DLR)

## Licenses
This software is licensed under the [Apache 2.0 License](https://github.com/dlr-eoc/ukis-metrics/blob/main/LICENSE).

Copyright (c) 2021 German Aerospace Center (DLR) * German Remote Sensing Data Center * Department: Geo-Risks and Civil Security

## Changelog
See [changelog](https://github.com/dlr-eoc/ukis-metrics/blob/main/CHANGELOG.md).


## What is UKIS?
The DLR project Environmental and Crisis Information System (the German abbreviation is UKIS, standing for [Umwelt- und Kriseninformationssysteme](https://www.dlr.de/eoc/en/desktopdefault.aspx/tabid-5413/10560_read-21914/) aims at harmonizing the development of information systems at the German Remote Sensing Data Center (DFD) and setting up a framework of modularized and generalized software components.

UKIS is intended to ease and standardize the process of setting up specific information systems and thus bridging the gap from EO product generation and information fusion to the delivery of products and information to end users.

Furthermore, the intention is to save and broaden know-how that was and is invested and earned in the development of information systems and components in several ongoing and future DFD projects.
