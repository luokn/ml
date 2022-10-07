#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @File  : utils.py
# @Time  : 2022/05/18 15:11:16
# @Author: Kun Luo
# @Email : olooook@outlook.com

from typing import List, Optional

import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics


def plot_roc_curve(
    title: str,
    y_truth: np.ndarray,
    y_score: np.ndarray,
    classes: List[str],
    out_file: Optional[str] = None,
    show_fig: bool = False,
):
    """
    Plot ROC curve for multi-class classification.

    Args:
        title (str): Title of the plot.
        y_truth (np.ndarray): Ground truth labels.
        y_score (np.ndarray): Predicted scores.
        classes (List[str]): List of class names.
        out_file (Optional[str], optional): Path to save the plot. Defaults to None. Defaults to None.
        show_fig (bool, optional): Whether to show the plot. Defaults to False.
    """
    plt.figure(figsize=[8, 8], dpi=200)
    plt.title(title)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel("FPR")
    plt.ylabel("TPR")

    # Plot the diagonal line.
    plt.plot([0, 1], [0, 1], "k-.", lw=1)

    # Plot the ROC curve for each class.
    fpr_all, tpr_all = [], []
    for i, name in enumerate(classes):
        fpr, tpr, _ = metrics.roc_curve(y_truth[:, i], y_score[:, i])
        fpr_all += [fpr]
        tpr_all += [tpr]
        auc = metrics.auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, label=f"ROC curve of class {name} (AUC = {auc:0.3f})")

    # Plot the micro-average ROC curve.
    micro_fpr, micro_tpr, _ = metrics.roc_curve(y_truth.reshape(-1), y_score.reshape(-1))
    micro_auc = metrics.auc(micro_fpr, micro_tpr)
    plt.plot(micro_fpr, micro_tpr, "--", lw=1, label=f"Micro-average ROC curve (AUC = {micro_auc:0.3f})")

    # Plot the macro-average ROC curve.
    macro_fpr = np.unique(np.concatenate(fpr_all))
    macro_tpr = sum((np.interp(macro_fpr, fpr, tpr) for fpr, tpr in zip(fpr_all, tpr_all))) / len(classes)
    macro_auc = metrics.auc(macro_fpr, macro_tpr)
    plt.plot(macro_fpr, macro_tpr, "--", lw=1, label=f"Macro-average ROC curve (AUC = {macro_auc:0.3f})")

    plt.legend()
    if out_file is not None:
        plt.savefig(out_file)
        print(f"Save ROC curve to {out_file}")
    if show_fig:
        plt.show()
