"""Script para entrenar el modelo"""
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from routers.aidtec.transformers import WineDatasetTransformer
from routers.aidtec.models import SerializableClassifier

def train_best_model(save_model: bool = False, save_encoder: bool = False) -> None:
    """Entrena el mejor modelo para el desafío"""

    # Aplicamos las transformaciones
    wt_train = WineDatasetTransformer(
        corregir_alcohol=True,
        corregir_densidad=True,
        shuffle=True,
        color_interactions=False,
        densidad_alcohol_interaction=True,
        remove_outliers=False,
        standardize=False,
        ratio_diox=True,
        rbf_diox=True,
        drop_columns=['color', 'year', 
                    'densidad', 'alcohol',
                    'dioxido de azufre libre'],
    )
    label_encoder = LabelEncoder()

    print("Cargando el dataset ...")
    train_df = pd.read_csv("datasets/aidtec_train.csv", index_col=0)
    X = train_df.drop(columns='calidad')
    y_transformed = label_encoder.fit_transform(train_df['calidad'])
    print('Transformando dataset ...')
    X_transformed = wt_train.fit_transform(X)
    y_transformed = y_transformed[X_transformed.index]

    rf = RandomForestClassifier(random_state=42,
                            criterion='gini',
                            n_estimators=900)

    rf_definitivo = SerializableClassifier(rf)

    print(X_transformed.head(5))

    print('Evaluando modelo con CV ...')
    scores = cross_val_score(
        rf_definitivo,
        X_transformed,
        y_transformed,
        cv=StratifiedKFold(n_splits=5),
        scoring='accuracy',
        n_jobs=-1,
        verbose=2
        )
    print('Accuracy media en CV:', scores.mean())   

    print('Entrenando el modelo ...')
    rf_definitivo.fit(X_transformed, y_transformed)

    if save_encoder:
        with open('models/wine_label_encoder.pkl', 'wb') as f:
            pickle.dump(label_encoder, f)

    if save_model:
        rf_definitivo.save('models/wine_random_forest_STM.pkl')
        print('Modelo guardado correctamente')

if __name__ == '__main__':
    train_best_model(save_encoder=True)