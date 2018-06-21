from sklearn.metrics import precision_recall_curve, recall_score, f1_score, average_precision_score
from sklearn.metrics import accuracy_score, classification_report, make_scorer, confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.metrics import precision_recall_fscore_support

from keras.callbacks import Callback

import numpy as np
import pandas as pd
import time


class RocAucEvaluation(Callback):

    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred, average='weighted')
            print(
                "\n ROC-AUC - epoch: %d - score: %.6f \n" %
                (epoch + 1, score))


def class_report(conf_mat):
    tp, fp, fn, tn = conf_mat.flatten()
    measures = {}
    measures['accuracy'] = (tp + tn) / (tp + fp + fn + tn)
    measures['specificity'] = tn / (tn + fp)        # (true negative rate)
    # (recall, true positive rate)
    measures['sensitivity'] = tp / (tp + fn)
    measures['precision'] = tp / (tp + fp)
    measures['f1score'] = 2 * tp / (2 * tp + fp + fn)
    return measures

def class_report_binary(y_test, y_scores):
    y_pred = (y_scores[:, 1] >= 0.50).astype(int)
    conf_mat = confusion_matrix(y_test, y_pred)
    measures = class_report(conf_mat)
    print("done")
    return measures

def class_report_multilabel(y_test, y_scores, verbose=True, combined=False):
    if isinstance(y_test, pd.core.frame.DataFrame):
        y_test = y_test.values

    #print(y_test.shape, y_scores.shape)
    y_pred = (y_scores >= 0.50).astype(int)
    measures = {}
    results_by_class = {}

    """
    fp = len(np.where((y_pred == 1) & (y_test == 0))[0])
    fn = len(np.where((y_pred == 0) & (y_test == 1))[0])
    tp = len(np.where((y_pred == 1) & (y_test == 1))[0])
    tn = len(np.where((y_pred == 0) & (y_test == 0))[0])

    measures['specificity'] = tn / (tn + fp)        # (true negative rate)
    measures['precision'] = tp / (tp + fp)
    """

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    precision = dict()
    recall = dict()
    average_precision = dict()

    n_classes = len(np.unique(np.where(y_test)[1]))
    pass_names = ["label_pa", "label_sb", "label_sleep"]
    for i, pass_name in zip(range(n_classes), pass_names):
        results_by_class[pass_name] = {}

        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        precision[i], recall[i], _ = precision_recall_curve(y_test[:, i],
                                                            y_scores[:, i])
        average_precision[i] = average_precision_score(
            y_test[:, i], y_scores[:, i])

        p, r, fscore, support = precision_recall_fscore_support(y_test[:, i], y_pred[:, i])
        acc = accuracy_score(y_test[:, i], y_pred[:, i])

        results_by_class[pass_name]["test_accuracy"] = acc
        results_by_class[pass_name]["test_roc_auc"] = roc_auc[i]
        results_by_class[pass_name]["test_avg_precision"] = average_precision[i]
        results_by_class[pass_name]["test_precision"] = p[1]
        results_by_class[pass_name]["test_recall"] = r[1]
        results_by_class[pass_name]["test_f1score"] = fscore[1]

        if verbose:
            print("ROC AUC for class %d: %.2f" % (i, roc_auc[i]))
            print("Precision for class %d: %.2f" % (i, average_precision[i]))

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_scores.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    test_roc_auc = roc_auc_score(y_test, y_scores, average='weighted')

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(
        y_test.ravel(), y_scores.ravel())
    average_precision["micro"] = average_precision_score(y_test, y_scores,
                                                         average="micro")

    average_precision["weighted"] = average_precision_score(y_test, y_scores,
                                                         average="weighted")

    average_precision["none"] = average_precision_score(y_test, y_scores,
                                                         average=None)

    p, r, fscore, support = precision_recall_fscore_support(y_test, y_pred, average='weighted')

    measures['test_accuracy'] = accuracy_score(y_test, y_pred)
    measures['test_roc_auc'] = test_roc_auc
    measures['weighted_recall'] = r
    measures['weighted_f1score'] = fscore
    measures['weighted_precision'] = p
    measures['weighted_avg_precision'] = average_precision["weighted"]

    if verbose:
        print(
            'Average precision score, micro-averaged over all classes: {0:0.2f}' .format(
                average_precision["weighted"]))
        print("Test score: %0.2f\n" % (test_roc_auc))

    return measures, results_by_class, tpr, fpr, roc_auc
