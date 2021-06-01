import numpy as np
import ukis_metrics.seg_metrics as segm

from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, jaccard_score, cohen_kappa_score


def test_tpfptnfn():
    shape = (256, 256, 3)
    # case all pixels belong to the positive class and are detected
    y_true = np.ones(shape)
    y_pred = np.ones(shape)
    valid_mask = np.ones(shape)
    tpfptnfn = segm.tpfptnfn(y_true, y_pred, valid_mask)
    assert (tpfptnfn["tp"] == y_true.size) and (tpfptnfn["fp"] == 0) and (tpfptnfn["tn"] == 0 and (tpfptnfn[
        "fn"] == 0))

    # case no pixel belongs to the positive class and none are detected (which is correct)
    y_true = np.zeros(shape)
    y_pred = np.zeros(shape)
    valid_mask = np.ones(shape)
    tpfptnfn = segm.tpfptnfn(y_true, y_pred, valid_mask)
    assert (tpfptnfn["tn"] == y_true.size) and (tpfptnfn["fp"] == 0) and (tpfptnfn["tp"] == 0 and (tpfptnfn[
                                                                                                       "fn"] == 0))

    # case all pixels belong to the positive class but none are detected
    y_true = np.ones(shape)
    y_pred = np.zeros(shape)
    valid_mask = np.ones(shape)
    tpfptnfn = segm.tpfptnfn(y_true, y_pred, valid_mask)
    assert (tpfptnfn["fn"] == y_true.size) and (tpfptnfn["fp"] == 0) and (tpfptnfn["tn"] == 0 and (tpfptnfn[
                                                                                                       "tp"] == 0))

    # case no pixels belong to the positive class but all are wrongly detected
    y_true = np.zeros(shape)
    y_pred = np.ones(shape)
    valid_mask = np.ones(shape)
    tpfptnfn = segm.tpfptnfn(y_true, y_pred, valid_mask)
    assert (tpfptnfn["fp"] == y_true.size) and (tpfptnfn["fn"] == 0) and (tpfptnfn["tn"] == 0 and (tpfptnfn[
                                                                                                       "tp"] == 0))

    # if x pixels are marked as non valid in the valid mask, the total sum of tpfptnfn must shrink by x
    y_true = np.zeros(shape)
    y_pred = np.ones(shape)
    valid_mask = np.ones(shape)
    x = np.random.randint(0, shape[0])
    y = np.random.randint(0, shape[1])
    z = np.random.randint(0, shape[2])
    valid_mask[:x, :y, :z] = 0
    tpfptnfn = segm.tpfptnfn(y_true, y_pred, valid_mask)

    assert np.sum(valid_mask) == np.sum([tpfptnfn["tp"], tpfptnfn["fp"], tpfptnfn["tn"], tpfptnfn["fn"]]) == tpfptnfn[
        "n_valid_pixel"]

def test_accuracy():
    # define reference mask (y_true), predictions (y_pred) and a valid pixel mask
    shape = (256, 256, 3)
    y_true = np.random.randint(0, 2, size=shape)
    y_pred = np.random.randint(0, 2, size=shape)
    valid_mask = np.random.randint(0, 2, size=shape)

    # we need to remove the invalid pixels form the array before they are handed over to sklearn
    y_true_sk = y_true[valid_mask == 1]
    y_pred_sk = y_pred[valid_mask == 1]

    tpfptnfn = segm.tpfptnfn(y_true, y_pred, valid_mask)
    acc = segm._accuracy(tpfptnfn["tp"], tpfptnfn["fp"], tpfptnfn["tn"], tpfptnfn["fn"])
    assert acc == accuracy_score(y_true_sk.flatten(), y_pred_sk.flatten())


def test_recall():
    # define reference mask (y_true), predictions (y_pred) and a valid pixel mask
    shape = (256, 256, 3)
    y_true = np.random.randint(0, 2, size=shape)
    y_pred = np.random.randint(0, 2, size=shape)
    valid_mask = np.random.randint(0, 2, size=shape)

    # we need to remove the invalid pixels form the array before they are handed over to sklearn
    y_true_sk = y_true[valid_mask == 1]
    y_pred_sk = y_pred[valid_mask == 1]
    tpfptnfn = segm.tpfptnfn(y_true, y_pred, valid_mask)
    assert segm._recall(tpfptnfn["tp"], tpfptnfn["fn"]) == recall_score(y_true_sk.flatten(), y_pred_sk.flatten())


def test_precision():
    # define reference mask (y_true), predictions (y_pred) and a valid pixel mask
    shape = (256, 256, 3)
    y_true = np.random.randint(0, 2, size=shape)
    y_pred = np.random.randint(0, 2, size=shape)
    valid_mask = np.random.randint(0, 2, size=shape)

    # we need to remove the invalid pixels form the array before they are handed over to sklearn
    y_true_sk = y_true[valid_mask == 1]
    y_pred_sk = y_pred[valid_mask == 1]
    tpfptnfn = segm.tpfptnfn(y_true, y_pred, valid_mask)
    assert segm._precision(tpfptnfn["tp"], tpfptnfn["fp"]) == precision_score(y_true_sk.flatten(), y_pred_sk.flatten())


def test_f1_score():
    # define reference mask (y_true), predictions (y_pred) and a valid pixel mask
    shape = (256, 256, 3)
    y_true = np.random.randint(0, 2, size=shape)
    y_pred = np.random.randint(0, 2, size=shape)
    valid_mask = np.random.randint(0, 2, size=shape)

    # we need to remove the invalid pixels form the array before they are handed over to sklearn
    y_true_sk = y_true[valid_mask == 1]
    y_pred_sk = y_pred[valid_mask == 1]
    tpfptnfn = segm.tpfptnfn(y_true, y_pred, valid_mask)
    rec = segm._recall(tpfptnfn["tp"], tpfptnfn["fn"])
    prec = segm._precision(tpfptnfn["tp"], tpfptnfn["fp"])
    assert np.round(segm._f1_score(rec, prec), decimals=6) == np.round(
        f1_score(y_true_sk.flatten(), y_pred_sk.flatten()), decimals=6
    )


def test_intersection_over_union():
    # define reference mask (y_true), predictions (y_pred) and a valid pixel mask
    shape = (256, 256, 3)
    y_true = np.random.randint(0, 2, size=shape)
    y_pred = np.random.randint(0, 2, size=shape)
    valid_mask = np.random.randint(0, 2, size=shape)

    # we need to remove the invalid pixels form the array before they are handed over to sklearn
    y_true_sk = y_true[valid_mask == 1]
    y_pred_sk = y_pred[valid_mask == 1]
    tpfptnfn = segm.tpfptnfn(y_true, y_pred, valid_mask)
    assert segm._intersection_over_union(
        tpfptnfn["tp"], tpfptnfn["fp"], tpfptnfn["fn"], tpfptnfn["tn"]
    ) == jaccard_score(y_true_sk.flatten(), y_pred_sk.flatten())


def test_kappa():
    # define reference mask (y_true), predictions (y_pred) and a valid pixel mask
    shape = (256, 256, 3)
    y_true = np.random.randint(0, 2, size=shape)
    y_pred = np.random.randint(0, 2, size=shape)
    valid_mask = np.random.randint(0, 2, size=shape)

    # we need to remove the invalid pixels form the array before they are handed over to sklearn
    y_true_sk = y_true[valid_mask == 1]
    y_pred_sk = y_pred[valid_mask == 1]
    tpfptnfn = segm.tpfptnfn(y_true, y_pred, valid_mask)
    assert np.round(segm._kappa(
        tpfptnfn["tp"], tpfptnfn["fp"], tpfptnfn["tn"], tpfptnfn["fn"], tpfptnfn["n_valid_pixel"]), decimals=6) == \
           np.round(cohen_kappa_score(y_true_sk.flatten(), y_pred_sk.flatten()), decimals=6)
