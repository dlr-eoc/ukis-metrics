"""Execution time benchmark of ukis-metrics against scikit-learn metrics

This script compares the execution time for ukis-metrics with sklearn metrics. Simply run the script to
evaluate an array representing a prediction and the corresponding reference mask and print the execution times.
"""


import ukis_metrics.seg_metrics as segm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, jaccard_score, cohen_kappa_score
import time


def gain(a, b):
    return np.round(b / a, decimals=2)


def format_string(name, own_metric, sk_metric, decimal):
    return "%s\t\t%.6f\t\t\t\t%.6f\t\t\t%.2f" % (
        name,
        np.round(own_metric, decimals=decimal),
        np.round(sk_metric, decimals=decimal),
        gain(own_metric, sk_metric),
    )


def xtime(mode, shape):
    """
    Measures the execution time of ukis_metrics compared to sklearn

    :param mode: label for metric to benchmark
    :param shape: shape of dummy array
    :return:
    """
    y_true = np.random.randint(0, 2, size=shape)
    y_pred = np.random.randint(0, 2, size=shape)
    valid_mask = np.random.randint(0, 2, size=shape)
    # own metric
    if mode == "acc":
        # ukis_metrics
        tic = time.perf_counter()
        tpfptnfn_val = segm.tpfptnfn(y_true, y_pred, valid_mask)
        segm._accuracy(tpfptnfn_val["tp"], tpfptnfn_val["fp"], tpfptnfn_val["tn"], tpfptnfn_val["fn"])
        d_own = time.perf_counter() - tic
        # sklearn
        # we need to remove the invalid pixels form the array before they are handed over to sklearn
        tic = time.perf_counter()
        y_true_sk = y_true[valid_mask == 1]
        y_pred_sk = y_pred[valid_mask == 1]
        accuracy_score(y_true_sk.flatten(), y_pred_sk.flatten())
        d_sk = time.perf_counter() - tic

    elif mode == "rec":
        # ukis metrics
        tic = time.perf_counter()
        tpfptnfn_val = segm.tpfptnfn(y_true, y_pred, valid_mask)
        segm._recall(tpfptnfn_val["tp"], tpfptnfn_val["fn"])
        d_own = time.perf_counter() - tic
        # sklearn
        tic = time.perf_counter()
        y_true_sk = y_true[valid_mask == 1]
        y_pred_sk = y_pred[valid_mask == 1]
        recall_score(y_true_sk.flatten(), y_pred_sk.flatten())
        d_sk = time.perf_counter() - tic

    elif mode == "pre":
        # ukis_metrics
        tic = time.perf_counter()
        tpfptnfn_val = segm.tpfptnfn(y_true, y_pred, valid_mask)
        segm._precision(tpfptnfn_val["tp"], tpfptnfn_val["fp"])
        d_own = time.perf_counter() - tic
        # sklearn
        tic = time.perf_counter()
        y_true_sk = y_true[valid_mask == 1]
        y_pred_sk = y_pred[valid_mask == 1]
        precision_score(y_true_sk.flatten(), y_pred_sk.flatten())
        d_sk = time.perf_counter() - tic

    elif mode == "f1":
        # ukis_metrics
        tic = time.perf_counter()
        tpfptnfn_val = segm.tpfptnfn(y_true, y_pred, valid_mask)
        rec = segm._recall(tpfptnfn_val["tp"], tpfptnfn_val["fn"])
        prec = segm._precision(tpfptnfn_val["tp"], tpfptnfn_val["fp"])
        segm._f1_score(rec, prec)
        d_own = time.perf_counter() - tic
        # sklearn
        tic = time.perf_counter()
        y_true_sk = y_true[valid_mask == 1]
        y_pred_sk = y_pred[valid_mask == 1]
        f1_score(y_true_sk.flatten(), y_pred_sk.flatten())
        d_sk = time.perf_counter() - tic

    elif mode == "iou":
        # ukis_metrics
        tic = time.perf_counter()
        tpfptnfn_val = segm.tpfptnfn(y_true, y_pred, valid_mask)
        segm._intersection_over_union(tpfptnfn_val["tp"], tpfptnfn_val["fp"], tpfptnfn_val["tn"], tpfptnfn_val["fn"])
        d_own = time.perf_counter() - tic
        # sklearn
        tic = time.perf_counter()
        y_true_sk = y_true[valid_mask == 1]
        y_pred_sk = y_pred[valid_mask == 1]
        jaccard_score(y_true_sk.flatten(), y_pred_sk.flatten())
        d_sk = time.perf_counter() - tic

    elif mode == "kap":
        # ukis_metrics
        tic = time.perf_counter()
        tpfptnfn_val = segm.tpfptnfn(y_true, y_pred, valid_mask)
        segm._kappa(
            tpfptnfn_val["tp"],
            tpfptnfn_val["fp"],
            tpfptnfn_val["tn"],
            tpfptnfn_val["fn"],
            tpfptnfn_val["n_valid_pixel"],
        )
        d_own = time.perf_counter() - tic
        # sklearn
        tic = time.perf_counter()
        y_true_sk = y_true[valid_mask == 1]
        y_pred_sk = y_pred[valid_mask == 1]
        cohen_kappa_score(y_true_sk.flatten(), y_pred_sk.flatten())
        d_sk = time.perf_counter() - tic

    return d_own, d_sk


if __name__ == "__main__":
    shape = (256, 256, 2)
    decimal = 6
    metrics = ["acc", "rec", "pre", "f1", "iou", "kap"]
    print(f"\nShape of array: {shape}\n")
    print("\t\t\t\t\t### Metrics execution time [s] ###\n")
    print("\t\tukis_metrics\t\t\t\tsklearn metrics\t\tspeed gain")

    for metric in metrics:
        d_own, d_sk = xtime(metric, shape)
        print(format_string(metric, d_own, d_sk, decimal))
