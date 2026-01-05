import numpy as np
import pandas as pd
import streamlit as st

from .model_utils import pred_threshold, dbscan_predict


def predecir(model, data, model_option):
    try:
        # Alinear columnas de entrada a lo esperado por el pipeline si es posible
        expected_cols = None
        if hasattr(model, 'feature_names_in_'):
            expected_cols = list(model.feature_names_in_)
        elif hasattr(model, 'named_steps'):
            prepro = model.named_steps.get('prepro_2_del') or model.named_steps.get('preprocessor') or model.named_steps.get('preprocessor_3_pca')
            if prepro is not None and hasattr(prepro, 'get_feature_names_out'):
                try:
                    expected_cols = list(prepro.get_feature_names_out())
                except Exception:
                    expected_cols = None

        if expected_cols:
            for col in expected_cols:
                if col not in data.columns:
                    data[col] = 0
            data = data[expected_cols]

        columnas_pca2=['componente1','componente2','componente3','componente4','componente5','componente6']

        if(model_option=='IForest'):
            scores = model.decision_function(data)
            predicciones=pred_threshold(scores, -0.10)
        if(model_option=='LOF'):
            scores = model.decision_function(data)
            threshold = np.percentile(scores, 100 * 0.2)
            predicciones = np.where(scores >= threshold,0,1)
        if(model_option=='OCSVM'):
            scores = None
            try:
                scores = model.decision_function(data)
            except Exception:
                scores = None
            if st.session_state.get('ocsvm_use_sign', True):
                pred_raw = model.predict(data)
                predicciones = np.where(pred_raw == 1, 0, 1)
            else:
                if scores is None:
                    predicciones = model.predict(data)
                    predicciones = np.where(predicciones == 1, 0, 1)
                else:
                    pct = float(st.session_state.get('ocsvm_percentile', 0.45))
                    threshold = np.percentile(scores, 100 * pct)
                    predicciones = np.where(scores >= threshold, 0, 1)
        if(model_option=='KMEANS'):
            scores=""
            predicciones=model.predict(data)
        if model_option == 'DBSCAN':
            scores=""
            preproc = model.named_steps['preprocessor_3_pca']
            X = preproc.transform(data)
            db = model.named_steps['dbscan']
            predicciones = dbscan_predict(db, X)
        if(model_option=='AUTOENCODER'):
            proba = None
            est = None
            try:
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(data)
                est = getattr(model, 'named_steps', {}).get('autoencoder_classifier', None)
                if proba is None and est is not None and hasattr(est, 'predict_proba'):
                    proba = est.predict_proba(data)
            except Exception:
                proba = None
            rec_err = None
            try:
                if hasattr(model, 'reconstruction_error'):
                    rec_err = model.reconstruction_error(data)
                elif est is not None and hasattr(est, 'reconstruction_error'):
                    rec_err = est.reconstruction_error(data)
            except Exception:
                rec_err = None
            normal_idx = 0
            thr = st.session_state.get('ae_normal_threshold', 0.60)
            rec_thr = st.session_state.get('ae_reconstruction_threshold', 0.02)
            if isinstance(proba, np.ndarray):
                pred_list = []
                for i, row in enumerate(proba):
                    is_normal_proba = row[normal_idx] >= thr
                    is_normal_recon = (rec_err is not None and i < len(rec_err) and rec_err[i] <= rec_thr)
                    if is_normal_proba and (rec_err is None or is_normal_recon):
                        pred_list.append(0)
                    else:
                        pred_list.append(int(np.argmax(row)))
                predicciones = np.array(pred_list, dtype=int)
                scores = row[normal_idx] if 'row' in locals() else proba[:, normal_idx]
            else:
                try:
                    predicciones = model.predict(data)
                except Exception:
                    predicciones = est.predict(data) if est is not None else np.zeros(len(data), dtype=int)
                scores = ""
        pp3 = None
        try:
            if model_option in ['IForest','OCSVM','LOF','DBSCAN']:
                transformer = None
                if hasattr(model, 'named_steps'):
                    transformer = model.named_steps.get('preprocessor_3_pca') or model.named_steps.get('prepro_3_pca')
                if transformer is None and hasattr(model, '__getitem__'):
                    transformer = model[0]
                if transformer is not None and hasattr(transformer, 'transform'):
                    pp3_arr = transformer.transform(data)
                    if isinstance(pp3_arr, np.ndarray) and pp3_arr.shape[1] == 6:
                        pp3 = pd.DataFrame(pp3_arr, columns=columnas_pca2)
        except Exception:
            pp3 = None
        if pp3 is None:
            pp3 = data.select_dtypes(include=[np.number]).copy()
        pp3['cluster'] = predicciones
        return pp3, predicciones, scores
    except Exception as e:
        st.error(f"Error en la predicción: {str(e)}")
        import traceback
        st.error(f"Detalles: {traceback.format_exc()}")
        return None, None, None


def mostrar_metricas(silhouette, calinski, davies):
    st.markdown("### 📊 Métricas Internas")
    cols = st.columns(3)
    with cols[0]:
        st.metric(
            "Silhouette Score",
            f"{silhouette:.3f}",
            delta="Bueno" if silhouette > 0.5 else "Regular"
        )
    with cols[1]:
        st.metric(
            "Calinski Score",
            f"{calinski:.3f}",
            delta="Bueno" if calinski > 1000 else "Regular"
        )
    with cols[2]:
        st.metric(
            "Davies Score",
            f"{davies:.3f}",
            delta="Bueno" if davies < 0.5 else "Regular"
        )
