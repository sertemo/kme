"""Script que recoge funciones auxiliares de todo tipo relacionadas con los datasets.
Serán funciones generales que se pueden aplicar a cualquier tipo de dataset
    """

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (balanced_accuracy_score,
                            precision_score,
                            recall_score,
                            roc_curve,
                            precision_recall_curve,
                            confusion_matrix,
                            f1_score,
                            auc,
                            matthews_corrcoef,
                            accuracy_score)

def codificar_labels(y_test:pd.DataFrame, labels_map:dict) -> np.ndarray:
    """Devuelve el dataset aplicando el labels_map

    Parameters
    ----------
    y_test : pd.DataFrame
        _description_
    labels_map : dict
        _description_

    Returns
    -------
    np.ndarray
        _description_
    """
    y_test_tic = y_test.replace(labels_map)
    return y_test_tic

def y_preds_to_csv(X_test_raw:pd.DataFrame, y_preds_raw:pd.DataFrame, target_col_name:str="y_preds") -> bytes:
    """Devuelve el DataFrame original X_test_raw con las predicciones y_preds
    fusionadas en última columna

    Parameters
    ----------
    X_test_raw : pd.DataFrame
        _description_
    y_preds_raw : pd.DataFrame
        _description_
        target_col_name : str
        El nombre de la columna de y_preds

    Returns
    -------
    bytes
        Retorna el dataset Para poder descargarlo desde Streamlit
    """
    # Cambiamos nombre de la columna de y_preds
    y_preds_raw.columns = [target_col_name]
    # Concatenamos los dataset
    X_download = pd.concat([X_test_raw, y_preds_raw], axis=1)
    output = X_download.to_csv().encode('utf-8')
    return output

def plotear_matriz_confusion(y_true:pd.DataFrame, y_pred:pd.DataFrame) -> plt.Figure:
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
    return plt

# Otra versión de la matriz de confusión
def plot_confmat(y_true:pd.DataFrame, y_preds:pd.DataFrame) -> plt.Figure:
    """Plotea la matriz de confusión"""

    confmat = confusion_matrix(y_true=y_true, y_pred=y_preds)
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')
    ax.xaxis.set_ticks_position('bottom')

    plt.xlabel('y_preds')
    plt.ylabel('y_test')
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

def computar_otras_metricas_binarias(y_true:pd.DataFrame, y_preds:np.ndarray) -> pd.Series:
    """Computa: precision, recall, f1 y la MCC y los devuelve en forma de dataframe

    Parameters
    ----------
    y_true : pd.DataFrame
        _description_
    y_preds : np.ndarray
        _description_

    Returns
    -------
    pd.DataFrame
        _description_
    """
    precision = precision_score(y_true=y_true, y_pred=y_preds)
    recall = recall_score(y_true=y_true, y_pred=y_preds)
    f1 = f1_score(y_true=y_true, y_pred=y_preds)
    mcc = matthews_corrcoef(y_true=y_true, y_pred=y_preds)

    df = pd.Series({
        'Precision': round(precision, 3),
        'Recall': round(recall, 3),
        'F1 score': round(f1, 3),
        'MCC': round(mcc, 3)
    })
    return df.rename("Valores")
