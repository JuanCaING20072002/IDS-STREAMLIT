import ast
import os
import joblib
import numpy as np
import pandas as pd
from collections import Counter

from sklearn.neighbors import NearestNeighbors
import streamlit as st

from .constants import (
    IFOREST_MODEL_PATHS,
    OCSVM_MODEL_PATH,
    KMEANS_MODEL_PATHS,
    AUTOENCODER_MODEL_PATH,
    KMEANS_TXT_PATH,
)


def load_model(model_path):
    try:
        absolute_path = os.path.abspath(model_path)
        st.write(f"Intentando cargar desde: {absolute_path}")
        if not os.path.exists(absolute_path):
            raise FileNotFoundError(f"El archivo {absolute_path} no se encuentra.")
        model = joblib.load(absolute_path)
        st.write(f"Tipo del modelo cargado: {type(model)}")
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo: {str(e)}")
        import traceback
        st.error(f"Detalles del error: {traceback.format_exc()}")
        return None


def dict_predict(path2):
    with open(path2, 'r') as f:
        raw = f.read()
    lista_tuplas = ast.literal_eval(raw)
    return dict(lista_tuplas)


def pred_threshold(score, threshold):
    score = pd.Series(score)
    Actual_pred = pd.DataFrame({'Pred': score})
    Actual_pred['Pred'] = np.where(Actual_pred['Pred'] <= threshold, 0, 1)
    return Actual_pred.reset_index(drop=True)


def dbscan_predict(dbscan, X_new):
    core_samples = dbscan.components_
    core_labels = dbscan.labels_[dbscan.core_sample_indices_]
    nn = NearestNeighbors(n_neighbors=1).fit(core_samples)
    dist, idx = nn.kneighbors(X_new)
    return np.where(dist.ravel() <= dbscan.eps, core_labels[idx.ravel()], -1)
