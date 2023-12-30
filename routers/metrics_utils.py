"""Script que recoge funciones auxiliares relacionadas con las métricas de los modelos.
Como por ejemplo computar accuracy, plotear gráficos de méticas, matrices de confusión etc.
Las funciones deberán ser de propósito general válidas para cualquier desafío
    """

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (balanced_accuracy_score,
                            precision_score,
                            recall_score,
                            roc_curve,
                            precision_recall_curve,
                            confusion_matrix,
                            f1_score,
                            auc,
                            matthews_corrcoef,
                            accuracy_score,
                            roc_auc_score)


def plotear_matriz_confusion(y_true:pd.DataFrame, y_pred:pd.DataFrame, ticks:list) -> plt.Figure:
    """Devuelve la figura de la matriz de confusión ploteada con Seaborn
    para utilizarla en streamlit

    Parameters
    ----------
    y_true : pd.DataFrame
        _description_
    y_pred : pd.DataFrame
        _description_

    Returns
    -------
    plt.Figure
        _description_
    """
    matriz_confusion:np.ndarray = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(matriz_confusion, annot=True, fmt='g')
    plt.xlabel('y_preds')
    plt.ylabel('y_test')
    plt.xticks([''] + ticks)
    return plt

# Otra versión de la matriz de confusión
def plot_confmat(y_true:pd.DataFrame, y_preds:pd.DataFrame, ticks:list) -> plt.Figure:
    """Plotea la matriz de confusión

    Parameters
    ----------
    y_true : pd.DataFrame
        _description_
    y_preds : pd.DataFrame
        _description_
    ticks : list
        la leyenda que quieres marcar en los ejes de la matriz

    Returns
    -------
    plt.Figure
        _description_
    """
    confmat = confusion_matrix(y_true=y_true, y_pred=y_preds)
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

    ax.set_xticklabels([''] + ticks);
    ax.set_yticklabels([''] + ticks);
    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(axis='both', which='major', labelsize=8)

    plt.xlabel('predicciones')
    plt.ylabel('real')
    plt.tight_layout()
    return plt

def plot_roc_auc(y_true:pd.DataFrame, y_probabilities:np.ndarray) -> plt.Figure:
    """Plotea la curva ROC AUC"""
    # Calcular TPR y FPR para varios umbrales
    fpr, tpr, thresholds = roc_curve(y_true, y_probabilities)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label='ROC Curve')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.fill_between(fpr, tpr, alpha=0.1, color='blue')
    plt.text(0.6, 0.3, 'AUC area', fontsize=12, color='blue')
    return plt

def plot_precision_recall_curve(y_true:pd.DataFrame, y_probabilities:np.ndarray) -> plt.Figure:
    precision, recall, thresholds = precision_recall_curve(y_true, y_probabilities)
    pr_auc = auc(recall, precision)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label='Precision-Recall curve (area = %0.2f)' % pr_auc)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Curva Precision-Recall')
    plt.legend(loc="best")
    return plt

def computar_otras_metricas(y_true:pd.DataFrame, y_preds:np.ndarray, labels:dict=None) -> tuple[float]:
    """Computa: precision, recall, f1 y la MCC y los devuelve en forma de dataframe.
    En ese orden

    Parameters
    ----------
    y_true : pd.DataFrame
        _description_
    y_preds : np.ndarray
        _description_
    labels : dict
        Solo para el caso de multiclass

    Returns
    -------
    tuple[float]
        _description_
    """
    precision = precision_score(y_true=y_true, y_pred=y_preds, labels=labels, average='weighted')
    recall = recall_score(y_true=y_true, y_pred=y_preds, labels=labels, average='weighted')
    f1 = f1_score(y_true=y_true, y_pred=y_preds, average='weighted')
    mcc = matthews_corrcoef(y_true=y_true, y_pred=y_preds)    
    return precision, recall, f1, mcc

def computar_accuracies(y_true:pd.DataFrame, y_preds:np.ndarray) -> tuple[float, float]:
    """Computa accuracy y balanced accuracy.
    Devuelve: (accuracy, balanced_accuracy)

    Parameters
    ----------
    y_true : pd.DataFrame
        _description_
    y_preds : np.ndarray
        _description_

    Returns
    -------
    tuple[float, float]
        _description_
    """
    acc = accuracy_score(y_true, y_preds)
    balanced_acc = balanced_accuracy_score(y_true, y_preds, adjusted=True)
    return acc, balanced_acc

def plot_roc_auc_multiclass(y_test, y_prob, labels_map:dict) -> None:
    y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
    #roc_auc_ovo = roc_auc_score(y_test_bin, y_prob, multi_class='ovo')
    n_classes = y_test_bin.shape[1]
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(10, 8))
    colors = iter(plt.cm.rainbow(np.linspace(0, 1, n_classes)))

    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                label='ROC curve de clase {0} (area = {1:0.2f})'
                ''.format(labels_map[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Ratio Falsos Positivos')
    plt.ylabel('Ratio Verdaderos Positivos')
    plt.title('Multi-class ROC')
    plt.legend(loc="best")
    return plt