import numpy as np
import sklearn.metrics as sk
from sklearn.preprocessing import binarize
from pprint import pprint

# from scripts.script_eval_data import plot_confusion_matrix_1
import matplotlib.pyplot as plt
import numpy as np
import itertools
import os

def plot_confusion_matrix_1(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):

    print('Using script')
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation="vertical")
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.2f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.2f}; misclass={:0.2f}'.format(
        accuracy, misclass))
#     plt.show()
    if not os.path.exists("./evaluate/"):
        os.mkdir("./evaluate")
    plt.savefig('./evaluate/confusion_matrix.png')




recall_level = 0.95
    
def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out


def fpr_and_fdr_at_recall(y_true, y_score, recall_level=recall_level,
                          pos_label=None):
    classes = np.unique(y_true)
    if (pos_label is None and
            not (np.array_equal(classes, [0, 1]) or
                     np.array_equal(classes, [-1, 1]) or
                     np.array_equal(classes, [0]) or
                     np.array_equal(classes, [-1]) or
                     np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps      # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)      # [last_ind::-1]
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))

    return fps[cutoff] / (np.sum(np.logical_not(y_true))), fps[cutoff]/(fps[cutoff] + tps[cutoff])


def show_performance(pos, neg, expected_ap=1 / (1 + 10.), method_name='Ours', recall_level=recall_level):
    '''
    :param pos: 1's class, class to detect, outliers, or wrongly predicted
    example scores from the baseline
    :param neg: 0's class scores generated by the baseline
    :param expected_ap: this is changed from the default for failure detection
    '''
    pos = np.array(pos).reshape((-1, 1))
    neg = np.array(neg).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[:len(pos)] += 1

    auroc = sk.roc_auc_score(labels, examples)
    aupr = sk.average_precision_score(labels, examples)
    fpr, fdr = fpr_and_fdr_at_recall(labels, examples, recall_level)

    print('\t\t\t' + method_name)
    print('FPR{:d}:\t\t\t{:.2f}'.format(int(100 * recall_level), 100 * fpr))
    print('AUROC:\t\t\t{:.2f}'.format(100 * auroc))
    print('AUPR:\t\t\t{:.2f}'.format(100 * aupr))
    # print('FDR{:d}:\t\t\t{:.2f}'.format(int(100 * recall_level), 100 * fdr))

def get_performance(pos, neg, expected_ap=1 / (1 + 10.), method_name='Ours', recall_level=recall_level):
    '''
    :param pos: 1's class, class to detect, outliers, or wrongly predicted
    example scores from the baseline
    :param neg: 0's class scores generated by the baseline
    :param expected_ap: this is changed from the default for failure detection
    '''
    pos = np.array(pos).reshape((-1, 1))
    neg = np.array(neg).reshape((-1, 1))

    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[:len(pos)] += 1

    fpr, tpr, thresholds = sk.roc_curve(labels, examples)
    print("list of thresholds:")
    # print(thresholds)
    print("---------------------")
    y_prob = examples.reshape(1, -1)
    y_preds = binarize(y_prob, -0.5)[0]
    y_preds = y_preds.astype(int)
    cm = sk.confusion_matrix(labels, y_preds)
    classes = ['other', 'handoff_human']
    plot_confusion_matrix_1(cm, normalize=True, target_names=classes, title="Confusion Matrix")
    
    # pprint(sk.classification_report(labels, y_preds))
    auroc = sk.roc_auc_score(labels, examples)
    aupr = sk.average_precision_score(labels, examples)

    fpr, fdr = fpr_and_fdr_at_recall(labels, examples, recall_level)

    print(examples.shape)
    print(labels.shape)

    return fpr, auroc, aupr


def show_performance_comparison(pos_base, neg_base, pos_ours, neg_ours, baseline_name='Baseline',
                                alternative_name='Ours', expected_ap=1 / (1 + 10.), recall_level=recall_level):
    '''
    :param pos_base: 1's class, class to detect, outliers, or wrongly predicted
    example scores from the baseline
    :param neg_base: 0's class scores generated by the baseline
    :param expected_ap: this is changed from the default for failure detection
    '''
    pos_base = np.array(pos_base).reshape((-1, 1))
    neg_base = np.array(neg_base).reshape((-1, 1))
    examples_base = np.squeeze(np.vstack((pos_base, neg_base)))
    labels_base = np.zeros(len(examples_base), dtype=np.int32)
    labels_base[:len(pos_base)] += 1

    auroc_base = sk.roc_auc_score(labels_base, examples_base)
    aupr_base = sk.average_precision_score(labels_base, examples_base)
    fpr_base, fdr_base = fpr_and_fdr_at_recall(labels_base, examples_base)

    del pos_base; del neg_base

    pos_ours = np.array(pos_ours).reshape((-1, 1))
    neg_ours = np.array(neg_ours).reshape((-1, 1))
    examples_ours = np.squeeze(np.vstack((pos_ours, neg_ours)))
    labels_ours = np.zeros(len(examples_ours), dtype=np.int32)
    labels_ours[:len(pos_ours)] += 1

    auroc_ours = sk.roc_auc_score(labels_ours, examples_ours)
    aupr_ours = sk.average_precision_score(labels_ours, examples_ours)
    fpr_ours, fdr_ours = fpr_and_fdr_at_recall(labels_ours, examples_ours)

    print('\t\t\t' + baseline_name + '\t' + alternative_name)
    print('FPR{:d}:\t\t\t{:.2f}\t\t{:.2f}'.format(
        int(100 * recall_level), 100 * fpr_base, 100 * fpr_ours))
    print('AUROC:\t\t\t{:.2f}\t\t{:.2f}'.format(
        100 * auroc_base, 100 * auroc_ours))
    print('AUPR:\t\t\t{:.2f}\t\t{:.2f}'.format(
        100 * aupr_base, 100 * aupr_ours))

    # print('FDR{:d}:\t\t\t{:.2f}\t\t{:.2f}'.format(
    #     int(100 * recall_level), 100 * fdr_base, 100 * fdr_ours))