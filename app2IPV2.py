import streamlit as st
import streamlit.components.v1 as components
import joblib
import pandas as pd
import numpy as np
from pyod.utils.utility import standardizer
from pyod.models.combination import average
import plotly.graph_objects as go
from streamlit_lottie import st_lottie
import requests
import sklearn
import sklearn.metrics as metrics
from sklearn.metrics import auc, silhouette_score, calinski_harabasz_score, davies_bouldin_score
from streamlit_option_menu import option_menu
import time
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from subprocess import PIPE, Popen
import shlex
import psutil
import os
import threading
import tempfile
import metricas
import pickle
from sklearn.base import BaseEstimator, ClassifierMixin
from del_columns import del_columns
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from collections import Counter
import ast
from sklearn.metrics import DistanceMetric
import sklearn.metrics._dist_metrics as _dm
import numpy as np
from sklearn.neighbors import NearestNeighbors
import psutil
import time
from autoencoder_classifier import AutoencoderClassifier

# >>> RUTAS DE MODELOS (globales, visibles para todo el módulo)
IFOREST_MODEL_PATHS = "model/iforest__EJERCICIO10_n_components6_fit.pkl"
OCSVM_MODEL_PATH    = "model/ocsvm__EJERCICIO10_n_components6_fit.pkl"
KMEANS_MODEL_PATHS  = "model/kmeans1__EJERCICIO10_n_components6_over.pkl"
AUTOENCODER_MODEL_PATH = "model/autoencoder.pkl"
KMEANS_TXT_PATH     = "model/list_kmeans_over.txt"

def load_model(model_path):
    try:
        # Cargar el modelo usando joblib
        model = joblib.load(model_path)
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

# IMPORTANTE: st.set_page_config debe ser la primera llamada a Streamlit en el script
st.set_page_config(
    page_title="Sistema IDS IoT - Detección de Intrusiones",
    page_icon="🛡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS
st.markdown(
    """
<style>
    body { color: #ffffff; font-family: 'Roboto', sans-serif; }
    .mi-caja label { color: black !important; }
    .mi-caja input { color: black !important; background-color: white !important; caret-color: black !important; }
    .main-header { font-family: 'Roboto', sans-serif; background: linear-gradient(90deg, #1E3D59 0%, #2E5077 100%); padding: 20px; border-radius: 10px; color: white; text-align: center; margin-bottom: 30px; }
    .metric-card { background: rgba(255, 255, 255, 0.1); padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); margin: 10px 0; transition: transform 0.3s ease; }
    .metric-card:hover { transform: translateY(-5px); }
    .success-animation { animation: fadeInOut 3s forwards; }
    @keyframes fadeInOut { 0% { opacity: 0; } 20% { opacity: 1; } 80% { opacity: 1; } 100% { opacity: 0; display: none; } }
    .stAlert { background-color: rgba(25, 25, 25, 0.5); color: white; border: none; padding: 1rem; border-radius: 10px; }
    .stButton > button { background: rgba(255, 255, 255, 0.1); border: none; color: #ffffff; padding: 10px 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1), 0 1px 3px rgba(0, 0, 0, 0.08); transition: all 0.3s ease; }
    .stButton > button:hover { background: rgba(255, 255, 255, 0.2); transform: translateY(-2px); box-shadow: 0 7px 14px rgba(0, 0, 0, 0.1), 0 3px 6px rgba(0, 0, 0, 0.08); }
    .stButton > button:active { transform: translateY(1px); box-shadow: 0 3px 8px rgba(0, 0, 0, 0.1), 0 1px 3px rgba(0, 0, 0, 0.08); }
    .dashboard-form { background: rgba(255, 255, 255, 0.1); padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); max-width: 600px; margin: auto; }
    .notification { padding: 10px; border-radius: 5px; margin-bottom: 10px; animation: fadeOut 2s forwards; animation-delay: 3.5s; }
    @keyframes fadeOut { from {opacity: 1;} to {opacity: 0; height: 0; padding: 0; margin: 0;} }
    .streamlit-expanderHeader, .stTextInput > div > div > input { color: #ffffff !important; background-color: rgba(255, 255, 255, 0.1) !important; }
    .stDataFrame { color: #ffffff; }
    .card { background: rgba(255, 255, 255, 0.1); padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); margin-bottom: 20px; }
    .carousel { display: flex; overflow-x: auto; scroll-snap-type: x mandatory; }
    .carousel-item { flex: none; scroll-snap-align: start; margin-right: 20px; }
</style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------- AUTENTICACIÓN ----------------------------
from pathlib import Path
import datetime as _dt
import auth

auth.init_db()

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.username = None

def _login_ui():
    logo_path = Path(__file__).parent / "img" / "logo.png"
    st.markdown("""
        <div style="text-align:center; margin-top: 1rem;">
            <h2>Acceso Administrador</h2>
            <p>Ingrese sus credenciales y el código de acceso.</p>
        </div>
    """, unsafe_allow_html=True)
    st.image(str(logo_path), width=160)

    tabs = st.tabs(["Iniciar sesión", "Crear Admin", "Acerca de"])

    with tabs[0]:
        c1, c2 = st.columns([2, 1])
        with c1:
            with st.form("login_form"):
                u = st.text_input("Usuario", key="login_user")
                p = st.text_input("Contraseña", type="password", key="login_pass")
                code = st.text_input("Código de acceso (6 dígitos)", key="login_code", max_chars=6)
                submitted = st.form_submit_button("Entrar")
            if submitted:
                ok, msg = auth.verify_login(u, p, code)
                if ok:
                    st.session_state.authenticated = True
                    st.session_state.username = u
                    st.success("Autenticación exitosa. Redirigiendo…")
                    st.rerun()
                else:
                    st.error(msg)
        with c2:
            st.markdown("**Código de acceso**")
            if st.button("Generar código", use_container_width=True):
                code, exp = auth.generate_code(ttl_seconds=300)
                st.session_state._last_code = code
                st.session_state._last_exp = exp
            code, exp = auth.get_current_code()
            if code and exp:
                remaining = int((exp - _dt.datetime.utcnow()).total_seconds())
                remaining = max(0, remaining)
                st.info(f"Código vigente: {code} | Expira en {remaining}s")
            else:
                st.warning("No hay código vigente. Genere uno para iniciar sesión.")

    with tabs[1]:
        admins = auth.get_admin_count()
        if admins == 0:
            st.info("No existe un administrador. Crea el primero.")
            with st.form("create_admin_form"):
                u = st.text_input("Usuario Admin")
                p1 = st.text_input("Contraseña", type="password")
                p2 = st.text_input("Confirmar contraseña", type="password")
                create = st.form_submit_button("Crear Administrador")
            if create:
                if p1 != p2:
                    st.error("Las contraseñas no coinciden")
                else:
                    ok, msg = auth.create_admin(u, p1)
                    if ok:
                        st.success(msg)
                    else:
                        st.error(msg)
        else:
            st.warning("Ya existe al menos un administrador. Inicia sesión para administrar usuarios.")

    with tabs[2]:
        st.markdown("""
        - Solo usuarios Administradores pueden acceder al IDS.
        - La autenticación requiere usuario, contraseña y un código de 6 dígitos con validez limitada.
        - Usa el botón "Generar código" para producir un código temporal (expira en 5 minutos).
        """)

if not st.session_state.authenticated:
    _login_ui()
    st.stop()


# ---------------------------------------------------- INICIO PAGE----------------------------------------------------------------------------------------------------------------------------
# Navegación mejorada
from pathlib import Path

with st.sidebar:
    logo_path = Path(__file__).parent / "img" / "logo.png"
    st.image(str(logo_path), width=150)
    if st.session_state.get("authenticated"):
        st.caption(f"Conectado como: {st.session_state.username}")
        if st.button("Cerrar sesión", key="logout_btn"):
            st.session_state.authenticated = False
            st.session_state.username = None
            st.experimental_rerun()
    selected = option_menu(
        "Navegación",
        ["Inicio", "Panel de Control", "Comparar Modelos"],
        icons=["house", "graph-up", "bar-chart"],
        menu_icon="cast",
        default_index=0
    )

    if selected == "Inicio":
        st.session_state.page = "home"
    elif selected == "Panel de Control":
        st.session_state.page = "dashboard"
    else:
        st.session_state.page = "compare"

def show_home_page():
    # Hero Section con imagen principal
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0; background: linear-gradient(135deg, #1E3D59 0%, #2E5077 100%); border-radius: 15px; margin-bottom: 2rem;">
        <h1 style="color: white; font-size: 3rem; margin-bottom: 1rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
            🛡️ Sistema de Detección de Intrusiones para IoT
        </h1>
        <h3 style="color: #B8D4E3; font-weight: 300; margin-bottom: 2rem;">
            Protegiendo el futuro digital con inteligencia artificial avanzada
        </h3>
    </div>
    """, unsafe_allow_html=True)

    # Descripción principal mejorada
    st.markdown("""
    <div style="background: rgba(255, 255, 255, 0.05); padding: 2rem; border-radius: 15px; 
                border-left: 4px solid #4CAF50; margin: 2rem 0;">
        <p style="font-size: 1.2rem; line-height: 1.8; text-align: justify;">
            Bienvenido al <strong>Sistema de Detección de Intrusiones en Redes IoT</strong>, 
            una solución avanzada diseñada para monitorear y detectar anomalías 
            en redes de IoT, brindando seguridad y confiabilidad sin precedentes. 
        </p>
        <p style="font-size: 1.1rem; line-height: 1.8; text-align: justify;">
            Nuestro sistema utiliza modelos de vanguardia en detección de anomalías, como 
            <strong>IForest</strong> (Isolation Forest), <strong>OCSVM</strong> (One Class Support Vector Machine),
            <strong>K-MEANS</strong> y <strong>Autoencoder multiclase</strong> para analizar patrones complejos 
            y alertar sobre posibles irregularidades en la red en tiempo real.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Sección de características principales con iconos
    st.markdown("---")
    st.markdown("""
    <h2 style="text-align: center; color: #4CAF50; margin: 3rem 0 2rem 0;">
        🎯 Características Principales del Sistema
    </h2>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="feature-card" style="background: rgba(76, 175, 80, 0.1); padding: 1.5rem; 
                    border-radius: 15px; text-align: center; height: 250px; 
                    border: 1px solid rgba(76, 175, 80, 0.3);">
            <div style="font-size: 3rem; margin-bottom: 1rem;">🔍</div>
            <h4 style="color: #4CAF50;">Detección en Tiempo Real</h4>
            <p style="font-size: 0.9rem;">Monitoreo continuo y análisis instantáneo de patrones de red</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card" style="background: rgba(33, 150, 243, 0.1); padding: 1.5rem; 
                    border-radius: 15px; text-align: center; height: 250px; 
                    border: 1px solid rgba(33, 150, 243, 0.3);">
            <div style="font-size: 3rem; margin-bottom: 1rem;">🧠</div>
            <h4 style="color: #2196F3;">IA Avanzada</h4>
            <p style="font-size: 0.9rem;">Algoritmos de machine learning de última generación</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card" style="background: rgba(255, 152, 0, 0.1); padding: 1.5rem; 
                    border-radius: 15px; text-align: center; height: 250px; 
                    border: 1px solid rgba(255, 152, 0, 0.3);">
            <div style="font-size: 3rem; margin-bottom: 1rem;">📊</div>
            <h4 style="color: #FF9800;">Análisis Predictivo</h4>
            <p style="font-size: 0.9rem;">Predicción de amenazas y comportamientos anómalos</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="feature-card" style="background: rgba(244, 67, 54, 0.1); padding: 1.5rem; 
                    border-radius: 15px; text-align: center; height: 250px; 
                    border: 1px solid rgba(244, 67, 54, 0.3);">
            <div style="font-size: 3rem; margin-bottom: 1rem;">🚫</div>
            <h4 style="color: #F44336;">Protección Activa</h4>
            <p style="font-size: 0.9rem;">Identificación y mitigación automática de amenazas</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Sección de interpretación de resultados mejorada
    st.markdown("---")
    st.markdown("""
    <h2 style="text-align: center; color: #FF9800; margin: 3rem 0 2rem 0;">
        💬 Interpretación de Resultados
    </h2>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="background: rgba(76, 175, 80, 0.1); padding: 2rem; border-radius: 15px; 
                    border-left: 4px solid #4CAF50; height: 350px;">
            <h4 style="color: #4CAF50; text-align: center; margin-bottom: 1.5rem;">
                🟢 Métricas de Validación
            </h4>
            <ul style="list-style-type: none; padding: 0;">
                <li style="margin-bottom: 1rem; padding: 0.5rem; background: rgba(255,255,255,0.1); border-radius: 8px;">
                    <strong>Silhouette Score:</strong> Mide la calidad de los clusters (ideal > 0.5)
                </li>
                <li style="margin-bottom: 1rem; padding: 0.5rem; background: rgba(255,255,255,0.1); border-radius: 8px;">
                    <strong>Calinski Score:</strong> Evalúa la separación entre clusters
                </li>
                <li style="padding: 0.5rem; background: rgba(255,255,255,0.1); border-radius: 8px;">
                    <strong>Davies Score:</strong> Indica la similitud dentro de clusters
                </li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: rgba(33, 150, 243, 0.1); padding: 2rem; border-radius: 15px; 
                    border-left: 4px solid #2196F3; height: 450px;">
            <h4 style="color: #2196F3; text-align: center; margin-bottom: 1.5rem;">
                🔰 Clasificación de Anomalías
            </h4>
            <div style="margin-bottom: 1rem; padding: 0.8rem; background: rgba(255,255,255,0.1); border-radius: 8px;">
                <strong style="color: #4CAF50;">Normal:</strong> Tráfico de red esperado
            </div>
            <div style="margin-bottom: 1rem; padding: 0.8rem; background: rgba(255,255,255,0.1); border-radius: 8px;">
                <strong style="color: #FF5252;">Anómalo:</strong> Patrones sospechosos
            </div>
            <div style="padding: 0.8rem; background: rgba(255,255,255,0.1); border-radius: 8px;">
                <strong>Multiclase:</strong>
                <ul style="margin-top: 0.5rem; padding-left: 1rem;">
                    <li>DDoS_TCP</li>
                    <li>DDoS_UDP</li>
                    <li>Reconnaissance</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background: rgba(255, 152, 0, 0.1); padding: 2rem; border-radius: 15px; 
                    border-left: 4px solid #FF9800; height: 350px;">
            <h4 style="color: #FF9800; text-align: center; margin-bottom: 1.5rem;">
                📈 Métricas de Rendimiento
            </h4>
            <ul style="list-style-type: none; padding: 0;">
                <li style="margin-bottom: 1rem; padding: 0.5rem; background: rgba(255,255,255,0.1); border-radius: 8px;">
                    <strong>CPU:</strong> % Consumo por segundo
                </li>
                <li style="margin-bottom: 1rem; padding: 0.5rem; background: rgba(255,255,255,0.1); border-radius: 8px;">
                    <strong>Memoria:</strong> % Uso de RAM
                </li>
                <li style="margin-bottom: 1rem; padding: 0.5rem; background: rgba(255,255,255,0.1); border-radius: 8px;">
                    <strong>Disco:</strong> % Uso de almacenamiento
                </li>
                <li style="padding: 0.5rem; background: rgba(255,255,255,0.1); border-radius: 8px;">
                    <strong>Tiempo:</strong> Duración en segundos
                </li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Sección de modelos mejorada
    st.markdown("---")
    st.markdown("""
    <h2 style="text-align: center; color: #9C27B0; margin: 3rem 0 2rem 0;">
        🔎 Modelos de Detección de Anomalías
    </h2>
    """, unsafe_allow_html=True)
    
    # Grid de modelos con diseño mejorado
    col1, col2 = st.columns(2)
    
    with col1:
        with st.expander("🔵 OCSVM (One Class Support Vector Machine)", expanded=False):
            st.markdown("""
            <div style="background: rgba(63, 81, 181, 0.1); padding: 1.5rem; border-radius: 10px; margin: 1rem 0;">
                <p style="text-align: justify; line-height: 1.6;">
                El modelo OCSVM es un algoritmo de detección de anomalías basado en Support Vector Machine (SVM), 
                que funciona creando una frontera en el espacio de características que agrupa la mayor parte 
                de los datos de entrenamiento como normales y considera como anomalías aquellas muestras que 
                queden fuera de esta región.
                </p>
                <div style="background: rgba(76, 175, 80, 0.2); padding: 1rem; border-radius: 8px; margin-top: 1rem;">
                    <strong style="color: #4CAF50;">✅ Ventajas:</strong> Efectivo para datos de alta dimensionalidad
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with st.expander("🟠 K-means", expanded=False):
            st.markdown("""
            <div style="background: rgba(255, 152, 0, 0.1); padding: 1.5rem; border-radius: 10px; margin: 1rem 0;">
                <p style="text-align: justify; line-height: 1.6;">
                El modelo K-means es un algoritmo de agrupamiento que tiene como objetivo dividir un conjunto 
                de datos en grupos que sean lo más similares posible internamente y los grupos sean entre sí 
                lo más distintos posible.
                </p>
                <div style="background: rgba(76, 175, 80, 0.2); padding: 1rem; border-radius: 8px; margin-top: 1rem;">
                    <strong style="color: #4CAF50;">✅ Ventajas:</strong> Rápido y eficiente para grandes datasets
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        with st.expander("🟢 IForest (Isolation Forest)", expanded=False):
            st.markdown("""
            <div style="background: rgba(76, 175, 80, 0.1); padding: 1.5rem; border-radius: 10px; margin: 1rem 0;">
                <p style="text-align: justify; line-height: 1.6;">
                El modelo IForest utiliza árboles de aislamiento para identificar puntos de datos anómalos. 
                Es eficiente y efectivo para detectar anomalías en grandes conjuntos de datos mediante el 
                principio de que las anomalías son más fáciles de aislar que los puntos normales.
                </p>
                <div style="background: rgba(76, 175, 80, 0.2); padding: 1rem; border-radius: 8px; margin-top: 1rem;">
                    <strong style="color: #4CAF50;">✅ Ventajas:</strong> Escalable y robusto contra outliers
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with st.expander("🔴 Autoencoder multiclase", expanded=False):
            st.markdown("""
            <div style="background: rgba(244, 67, 54, 0.1); padding: 1.5rem; border-radius: 10px; margin: 1rem 0;">
                <p style="text-align: justify; line-height: 1.6;">
                Este modelo combina un autoencoder tradicional con una capa de clasificación supervisada, 
                formando un único grafo computacional que optimiza simultáneamente la reconstrucción de 
                entrada y la predicción de su etiqueta.
                </p>
                <div style="background: rgba(76, 175, 80, 0.2); padding: 1rem; border-radius: 8px; margin-top: 1rem;">
                    <strong style="color: #4CAF50;">✅ Ventajas:</strong> Combina detección y clasificación
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Guía de uso mejorada
    st.markdown("---")
    st.markdown("""
    <h2 style="text-align: center; color: #E91E63; margin: 3rem 0 2rem 0;">
        📖 Guía de Uso del Sistema
    </h2>
    """, unsafe_allow_html=True)
    
    # Timeline de pasos
    steps = [
        {"icon": "🔌", "title": "Seleccionar Modelo", "desc": "Elige el algoritmo de detección más adecuado", "color": "#2196F3"},
        {"icon": "📡", "title": "Capturar Paquetes", "desc": "Inicia la captura de tráfico de red", "color": "#4CAF50"},
        {"icon": "🖱️", "title": "Realizar Predicción", "desc": "Ejecuta el análisis con un solo clic", "color": "#FF9800"},
        {"icon": "📊", "title": "Visualizar Resultados", "desc": "Examina gráficos interactivos y métricas", "color": "#9C27B0"},
        {"icon": "💾", "title": "Guardar Resultados", "desc": "Exporta los datos en formato CSV", "color": "#F44336"}
    ]
    
    for i, step in enumerate(steps, 1):
        st.markdown(f"""
        <div style="display: flex; align-items: center; margin: 1.5rem 0; 
                    background: rgba(255, 255, 255, 0.05); padding: 1.5rem; border-radius: 15px;
                    border-left: 4px solid {step['color']};">
            <div style="font-size: 3rem; margin-right: 2rem; min-width: 80px; text-align: center;">
                {step['icon']}
            </div>
            <div>
                <h4 style="color: {step['color']}; margin-bottom: 0.5rem;">
                    Paso {i}: {step['title']}
                </h4>
                <p style="margin: 0; font-size: 1.1rem; opacity: 0.9;">
                    {step['desc']}
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Sección de tipos de resultados mejorada
    st.markdown("---")
    st.markdown("""
    <h2 style="text-align: center; color: #795548; margin: 3rem 0 2rem 0;">
        📊 Tipos de Resultados y Recomendaciones
    </h2>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #4CAF50 0%, #8BC34A 100%); 
                    padding: 2rem; border-radius: 15px; height: 465px; color: white;">
            <div style="text-align: center; font-size: 4rem; margin-bottom: 1rem;">✅</div>
            <h4 style="text-align: center; margin-bottom: 1.5rem;">Resultados Normales</h4>
            <p style="text-align: justify; line-height: 1.6; margin-bottom: 1.5rem;">
                Los datos han sido clasificados como normales. No se han detectado anomalías significativas 
                en el tráfico de red analizado.
            </p>
            <div style="background: rgba(255,255,255,0.2); padding: 1rem; border-radius: 8px;">
                <strong>💡 Recomendación:</strong><br>
                Continúa monitoreando la red regularmente para mantener la seguridad.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #FF9800 0%, #FFC107 100%); 
                    padding: 2rem; border-radius: 15px; height: 465px; color: white;">
            <div style="text-align: center; font-size: 4rem; margin-bottom: 1rem;">⚠️</div>
            <h4 style="text-align: center; margin-bottom: 1.5rem;">Resultados Anómalos</h4>
            <p style="text-align: justify; line-height: 1.6; margin-bottom: 1.5rem;">
                Se han detectado patrones anómalos en la red. Esto puede indicar posibles amenazas 
                o comportamientos inusuales que requieren atención.
            </p>
            <div style="background: rgba(255,255,255,0.2); padding: 1rem; border-radius: 8px;">
                <strong>🔍 Recomendación:</strong><br>
                Investiga las anomalías detectadas y toma medidas preventivas inmediatas.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #F44336 0%, #E91E63 100%); 
                    padding: 2rem; border-radius: 15px; height: 465px; color: white; overflow-y: auto;">
            <div style="text-align: center; font-size: 4rem; margin-bottom: 1rem;">🚨</div>
            <h4 style="text-align: center; margin-bottom: 1.5rem;">Ataques Específicos</h4>
            <!-- Tarjeta DDoS_TCP -->
            <div style="background:#ff5c5c;padding:1rem;border-radius:1rem;margin-bottom:1rem;color:white;">
                <strong>🔴 DDoS_TCP:</strong>
                <ul style="font-size: 0.9rem; margin-top: 0.5rem;">
                    <li>Activa SYN cookies</li>
                    <li>Configura límites de tasa</li>
                    <li>Usa ACL para bloquear IPs</li>
                </ul>
            </div>
            <!-- Tarjeta DDoS_UDP -->
            <div style="background:#ffb300;padding:1rem;border-radius:1rem;margin-bottom:1rem;color:white;">
                <strong>🟠 DDoS_UDP:</strong>
                <ul style="font-size: 0.9rem; margin-top: 0.5rem;">
                    <li>Permite solo puertos necesarios</li>
                    <li>Aplica límites de tasa UDP</li>
                    <li>Usa null routing</li>
                </ul>
            </div>
            <!-- Tarjeta Reconnaissance -->
            <div style="background:#2196f3;padding:1rem;border-radius:1rem;color:white;">
                <strong>🔵 Reconnaissance:</strong>
                <ul style="font-size: 0.9rem; margin-top: 0.5rem;">
                    <li>Restringe respuestas ICMP</li>
                    <li>Expón solo puertos necesarios</li>
                    <li>Configura límites anti-scan</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Call-to-action mejorado
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; background: linear-gradient(135deg, #1E3D59 0%, #2E5077 100%); 
                padding: 3rem; border-radius: 20px; margin: 3rem 0;">
        <h3 style="color: white; margin-bottom: 1.5rem;">¿Listo para comenzar?</h3>
        <p style="color: #B8D4E3; font-size: 1.2rem; margin-bottom: 2rem;">
            Dirígete al Panel de Control para empezar a proteger tu red IoT
        </p>
        <div style="font-size: 2rem;">🚀</div>
    </div>
    """, unsafe_allow_html=True)

#---------------------------------------------------- DASHBOARD ----------------------------------------------------------------------------------------------------------------------------
# DistanceMetric.get_metric("euclidean") devuelve un objeto cuya clase interna
# es la que Python debe encontrar al desempaquetar el pickle.
_dm.EuclideanDistance = DistanceMetric.get_metric("euclidean").__class__


 # RUTAS DE LOS MODELOS

def show_dashboard_page():
    st.title('❇ Panel de Control de Detección')

    # Inicializa el flag una sola vez
    if 'prediccion_realizada' not in st.session_state:
        st.session_state['prediccion_realizada'] = False

    model_path = "model/lof_1_6_1_EJERCICIO10_n_components6_fit_over.pkl"
    loaded_object = joblib.load(model_path)
    print(f"Tipo del objeto cargado: {type(loaded_object)}")
    print(f"Contenido: {loaded_object}")
    
    model_path = "model/iforest__EJERCICIO10_n_components6_fit.pkl"
    loaded_object = joblib.load(model_path)
    print(f"Tipo del objeto cargado: {type(loaded_object)}")
    print(f"Contenido: {loaded_object}")
    
    model_path = "model/kmeans1_6_1__EJERCICIO10_n_components6_over.pkl"
    loaded_object = joblib.load(model_path)
    print(f"Tipo del objeto cargado: {type(loaded_object)}")
    print(f"Contenido: {loaded_object}")
    
    model_path = "model/dbscan_1_6_1_EJERCICIO10_n_components6_fit_over.pkl"
    loaded_object = joblib.load(model_path)
    print(f"Tipo del objeto cargado: {type(loaded_object)}")
    print(f"Contenido: {loaded_object}")
    
    model_path = "model/ocsvm__EJERCICIO10_n_components6_fit.pkl"
    loaded_object = joblib.load(model_path)
    print(f"Tipo del objeto cargado: {type(loaded_object)}")
    print(f"Contenido: {loaded_object}")
    

    # Rutas de modelos permitidos (alineados con el Panel de Control)
    # IFOREST_MODEL_PATHS = "model/iforest__EJERCICIO10_n_components6_fit.pkl"
    # OCSVM_MODEL_PATH = "model/ocsvm__EJERCICIO10_n_components6_fit.pkl"
    # KMEANS_MODEL_PATHS = "model/kmeans1__EJERCICIO10_n_components6_over.pkl"
    # AUTOENCODER_MODEL_PATH = "model/autoencoder.pkl"
    # KMEANS_TXT_PATH = "model/list_kmeans_over.txt"


    # Inicializa el flag una sola vez
    if 'prediccion_realizada' not in st.session_state:
        st.session_state['prediccion_realizada'] = False

    # Estado para captura automática
    if 'auto_capture' not in st.session_state:
        st.session_state.auto_capture = {
            'active': False,
            'is_capturing': False,
            'stop_requested': False,
            'interval_sec': 300,
            'packets_per_capture': 100,
            'duration_sec': 0,  # 0 = sin límite por tiempo
            'next_run_ts': None,
            'last_cycle_result': None,
        }

    # Usar rutas globales: IFOREST_MODEL_PATHS, OCSVM_MODEL_PATH, KMEANS_MODEL_PATHS, AUTOENCODER_MODEL_PATH


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

    col1, col2 = st.columns([1, 2])

    # Columna derecha: bienvenida o resumen tras predicción
    with col2:
        if 'prediccion_realizada' not in st.session_state or not st.session_state['prediccion_realizada']:
            st.markdown("## 👋 Bienvenido al Panel IDS")
            st.info("Aquí verás el resumen y visualización de los resultados una vez realices una predicción. Selecciona el modelo y las opciones en el panel de la izquierda, luego captura paquetes y realiza la predicción.")
        else:
            st.markdown("## 📄 Resumen de la Predicción")
            resumen = st.session_state['resumen_prediccion']
            st.info(resumen['texto'])
            import plotly.express as px
            pie_fig = px.pie(
                names=resumen['labels'],
                values=resumen['values'],
                color=resumen['labels'],
                color_discrete_map={
                    'Normal': '#00C853',
                    'Anómalo': '#FF5252',
                    'DDOS_TCP': '#FF5252',
                    'DDOS_UDP': '#FFA726',
                    'Reconnaissance': '#42A5F5'
                },
                title="Distribución porcentual de clases"
            )
            pie_fig.update_traces(textinfo='percent+label')
            st.plotly_chart(pie_fig, use_container_width=True)
    
    # Interfas Usuario Panel de Control
    with col1:
        st.markdown("### 🖥 Panel de Control")
        model_option = st.selectbox(
            "📌 Seleccione el modelo de detección",
            ('IForest','OCSVM','KMEANS','AUTOENCODER'),
            help="Cada modelo utiliza diferentes técnicas para detectar anomalías"
        )
        metric_option = st.selectbox(
            "📌 Seleccione el tipo de metrica",
            ('Externas', 'Internas'),
            help="Cada modelo utiliza diferentes técnicas para detectar anomalías"
        )

        # ---------------- Captura automática ----------------
        st.title(" ⚙️ Captura automática ")
        col_auto_a, col_auto_b = st.columns(2)
        with col_auto_a:
            _default_dur_min = max(1, int(st.session_state.auto_capture['duration_sec'] // 60))
            _val_dur_min = st.number_input(
                'Duración de captura (minutos)', min_value=1, max_value=1440, value=_default_dur_min, step=1,
                help='La captura automática finalizará cuando se cumpla este tiempo'
            )
            st.session_state.auto_capture['duration_sec'] = int(_val_dur_min * 60)
        with col_auto_b:
            _default_minutes = max(1, int(st.session_state.auto_capture['interval_sec'] // 60))
            _val_minutes = st.number_input(
                'Intervalo entre capturas (minutos)', min_value=1, max_value=1440, value=_default_minutes, step=1,
                help='Pausa (en minutos) entre el fin de una captura y el inicio de la siguiente'
            )
            st.session_state.auto_capture['interval_sec'] = int(_val_minutes * 60)
            auto_status = '🟢 Activa' if st.session_state.auto_capture['active'] else '🔴 Detenida'
            st.metric('Estado', auto_status)

        col_btn_a, col_btn_b = st.columns(2)
        with col_btn_a:
            if st.button('▶️ Iniciar automática', use_container_width=True, disabled=st.session_state.auto_capture['active']):
                st.session_state.auto_capture['active'] = True
                st.session_state.auto_capture['stop_requested'] = False
                # Ejecutar inmediatamente la primera
                st.session_state.auto_capture['next_run_ts'] = time.time()
                st.rerun()
        with col_btn_b:
            if st.button('⏹️ Detener automática', use_container_width=True, disabled=not st.session_state.auto_capture['active']):
                st.session_state.auto_capture['stop_requested'] = True
                st.session_state.auto_capture['active'] = False
                st.session_state.auto_capture['next_run_ts'] = None

        # Countdown/estado próximo ciclo
        if st.session_state.auto_capture['active'] and not st.session_state.auto_capture['is_capturing']:
            if st.session_state.auto_capture['next_run_ts'] is not None:
                remaining = max(0, int(st.session_state.auto_capture['next_run_ts'] - time.time()))
                if remaining > 0:
                    m, s = divmod(remaining, 60)
                    st.info(f"⏳ Próxima captura en {m:02d}:{s:02d} min:s")
                    # Refrescar para avanzar el conteo
                    time.sleep(1)
                    st.rerun()
        
        
        # Función para procesar los paquetes capturados
        # ----------------------------------------------
        # Configuración del comando para Tshark
        comm_arg =  (
    "sudo /usr/local/bin/tshark "
    "-i eth0 -i wlan0 "
    "-l "
    "-Y 'eth.type != 0x8899' "
    # filtro de captura:
    "-f \"tcp or udp or arp\" "
    "-f \"arp or icmp or (udp and not port 53 and not port 5353) or (tcp and not port 443)\" "
    "-T fields -E separator=/t "
    "-e frame.time_delta "
    "-e _ws.col.Protocol "
    "-e ip.src -e ip.dst "
    "-e arp.src.proto_ipv4 -e arp.dst.proto_ipv4 "
    "-e ipv6.src -e ipv6.dst "
    "-e eth.src -e eth.dst "
    "-e tcp.srcport -e tcp.dstport "
    "-e udp.srcport -e udp.dstport "
    "-e frame.len -e udp.length "
    "-e ip.ttl -e icmp.type "
    "-e ip.dsfield.dscp -e ip.flags.rb -e ip.flags.df -e ip.flags.mf "
    "-e tcp.flags.res -e tcp.flags.ns -e tcp.flags.cwr -e tcp.flags.ecn "
    "-e tcp.flags.urg -e tcp.flags.ack -e tcp.flags.push -e tcp.flags.reset "
    "-e tcp.flags.syn -e tcp.flags.fin -e ip.version "
    "-e frame.time_epoch"
)

        comm_arg = shlex.split(comm_arg)

        # Función para capturar paquetes
        def capturing_packets(comm_arg):
            process = Popen(comm_arg, stdout=PIPE, stderr=PIPE, text=True)
            return process

        # Función para transformar los paquetes en el formato correcto
        def type_packet(packet_):
            if len(packet_) < 34:  # Rellenar con valores vacíos si falta información
                packet_ += [''] * (34 - len(packet_))

            
            if packet_[1] == 'TCP' or packet_[1] == 'SSH' or packet_[1] == 'SSHv2' or packet_[1] == 'SSHv2' or packet_[1] == 'TLSv1.2' or packet_[1] == 'HTTP':
                tcp_src = packet_[10]  # campo tcp.srcport
                tcp_dst = packet_[11]  # campo tcp.dstport
            elif packet_[1] == 'UDP' or packet_[1]=='DNS' or packet_[1]=='BROWSER' or packet_[1]=='DHCP' :
                tcp_src = packet_[12]  # reasignación de udp.srcport a tcp.srcport
                tcp_dst = packet_[13]  # reasignación de udp.dstport a tcp.dstport
            else:
                tcp_src = ''
                tcp_dst = ''

                
            if packet_[1] == 'ARP':
                packet_[2] = packet_[4]  # ip.src <- arp.src.proto_ipv4
                packet_[3] = packet_[5]  # ip.dst <- arp.dst.proto_ipv4

            # Mapeo de IPv6 al campo IP src y dst
            if packet_[32] == '6':
                packet_[2] = packet_[6]  # ip.src <- ipv6.src
                packet_[3] = packet_[7]  # ip.dst <- ipv6.dst

 

            # Reemplazo de comas por puntos en ciertos campos
            for index in [16, 18, 19, 20, 21]:
                if packet_[index]:
                    packet_[index] = packet_[index].replace(',', '.')

            # Estructura del paquete ordenado
            #ordered_packet = [
                #packet_[0], packet_[1],tcp_src, tcp_dst, packet_[10], packet_[11], packet_[14],
                #packet_[15], packet_[16], packet_[17], packet_[18], packet_[19], packet_[20], packet_[21],
                #packet_[22], packet_[23], packet_[24], packet_[25], packet_[26], packet_[27], packet_[28],
                #packet_[29], packet_[30], packet_[31], packet_[32]
            #]

            ordered_packet = [
                packet_[0], packet_[1], packet_[2],packet_[3],tcp_src, tcp_dst,packet_[14], packet_[15],
                packet_[16], packet_[17], packet_[18], packet_[19], packet_[20], packet_[21], packet_[22],
                packet_[23], packet_[24], packet_[25], packet_[26], packet_[27], packet_[28], packet_[29],
                packet_[30], packet_[31], packet_[33]
            ]


            fieldnames = [
                'delta_time', 'protocols','ip_src','ip_dst','port_src', 'port_dst', 'frame_len',
                'udp_len', 'ip_ttl', 'icmp_type', 'tos', 'ip_flags_rb', 'ip_flags_df', 'ip_flags_mf',
                'tcp_flags_res', 'tcp_flags_ns', 'tcp_flags_cwr', 'tcp_flags_ecn', 'tcp_flags_urg',
                'tcp_flags_ack', 'tcp_flags_push', 'tcp_flags_reset', 'tcp_flags_syn', 'tcp_flags_fin','epoch_time'
            ]

            return {fieldnames[i]: [ordered_packet[i]] for i in range(len(fieldnames))}

        # Función para agregar los paquetes al DataFrame
        def packet_df(type_packet, df):
            df_temp = pd.DataFrame(type_packet)
            if list(df_temp.columns) == ['value']:
                return df  # Ignorar tablas value
            if df is None:
                df = df_temp
            else:
                df = pd.concat([df, df_temp], axis=0, ignore_index=True)
            # Si después de concatenar solo queda la columna value, devolver None
            if list(df.columns) == ['value']:
                return None
            return df
        
        def predecir(model, data, model_option):
            try:
                
                
                # Obtener las columnas desde el preprocesador
                columns = model[0].named_steps['prepro_2_del']\
                  .named_steps['prepro_1_num_cat']\
                  .get_feature_names_out().tolist()

                # Reemplazar del_columns en el pipeline con la versión correcta
                model[0].named_steps['prepro_2_del'].named_steps['del_columns'] = del_columns(columns)
                # Asegurarse de que las columnas del DataFrame coincidan con las esperadas por el modelo
                columnas=['number_delta_time',	'numbertcp_srcport',	'numbertcp_dstport',	'numberframe_len',	'numberudp_length',	'numberip_ttl',	'numbertos',	'numberip_flags_df',	'numberip_flags_mf',	'numbertcp_flags_ns',	'number_tcp_flags_syn','category__protocols'	]
                
                columnas_pca2=['componente1','componente2','componente3','componente4','componente5','componente6']

                #'category__protocols'
                if(model_option=='IForest'):
                    scores = model.decision_function(data)
                    predicciones=pred_threshold(scores, -0.10)
                    
                if(model_option=='LOF'):
                    scores = model.decision_function(data)
                    threshold = np.percentile(scores, 100 * 0.2)
                    st.write(threshold)
                    predicciones = np.where(scores >= threshold,0,1)

                if(model_option=='OCSVM'):
                    scores = model.decision_function(data)
                    threshold = np.percentile(scores, 100 * 0.45)
                    st.write(threshold)
                    predicciones = np.where(scores >= threshold,0,1)
                    #predicciones1=model.predict(data)
                    #predicciones=np.where(predicciones1 == 1, 0, 1)
                    #scores = model.decision_function(data)
                    #predicciones=pred_threshold(scores, 16000.512)
                if(model_option=='KMEANS'):
                    scores=""
                    predicciones=model.predict(data)
                    
                #if(model_option=='DBSCAN'):
                    #preproc = model.named_steps['preprocessor_3_pca']
                    #X = preproc.transform(data) # → pp3
                    #est = model.named_steps['dbscan']
                    #predicciones = est.fit_predict(X)
                    
                if model_option == 'DBSCAN':
                # 1) Transformación previa (PCA, escalado, etc.)
                    scores=""
                    preproc = model.named_steps['preprocessor_3_pca']
                    X = preproc.transform(data)  # → pp3

                    # 2) DBSCAN ya entrenado (lo cargaste con joblib)
                    db = model.named_steps['dbscan']

                    # 3) Asignación de etiquetas sin refit
                    predicciones = dbscan_predict(db, X)

                if(model_option=='AUTOENCODER'):
                    print("Aqui estan tus pesos: ", model.named_steps['autoencoder_classifier'].model.get_weights()[0].shape)
                    predicciones=model.predict(data)
                    scores=""
                                
                pp3 = model[0].transform(data)  # Transformar los datos para el modelo IForest
                
                pp3 = pd.DataFrame(pp3, columns=columnas_pca2)  # Convertir a DataFrame
                
                # Añadir la columna de cluster al DataFrame
                pp3['cluster'] = predicciones
                
                
                return pp3, predicciones,scores
                
            except Exception as e:
                st.error(f"Error en la predicción: {str(e)}")
                import traceback
                st.error(f"Detalles: {traceback.format_exc()}")
                return None, None, None
        
        def mostrar_metricas(silhouette, calinski, davies):
          
           # Mostrar métricas en cards
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
                            
                     

        # Función para guardar los paquetes en CSV
        def save_packet_csv(df, path):
            df.to_csv(path, index=False)
            st.success(f"✅ Archivo guardado en {path}")
            
        # Auxiliar: ejecutar predicción y mostrar resultado (manual/automática)
        def run_prediction_on_data(data_df):
            try:
                model = None
                if model_option == 'IForest':
                    model = load_model(IFOREST_MODEL_PATHS)
                elif model_option == 'OCSVM':
                    model = load_model(OCSVM_MODEL_PATH)
                elif model_option == 'KMEANS':
                    model = load_model(KMEANS_MODEL_PATHS)
                elif model_option == 'AUTOENCODER':
                    model = load_model(AUTOENCODER_MODEL_PATH)
                if model is None:
                    st.error('No se pudo cargar el modelo seleccionado')
                    return

                lista = ['ip_ttl','tos','ip_flags_rb','ip_flags_df','ip_flags_mf']
                if set(lista).issubset(set(data_df.columns)):
                    data_df[lista] = data_df[lista].astype('str')
                    data_df.ip_ttl = data_df.ip_ttl.str.replace(',','.')
                    data_df.tos = data_df.tos.str.replace(',','.')
                    data_df.ip_flags_rb = data_df.ip_flags_rb.str.replace(',','.')
                    data_df.ip_flags_df = data_df.ip_flags_df.str.replace(',','.')
                    data_df.ip_flags_mf = data_df.ip_flags_mf.str.replace(',','.')
                data_df = data_df.replace('', np.nan).fillna(0)

                start_time = time.time()
                cpu_usage = psutil.cpu_percent(1)
                mem_usage = psutil.virtual_memory().percent
                disco2 = psutil.disk_usage('/')

                pp3, y_pred, scores = predecir(model, data_df, model_option)
                if pp3 is None or y_pred is None:
                    st.error('Error en la predicción')
                    return

                # Conteos y etiquetas
                if model_option == 'KMEANS':
                    path2 = KMEANS_TXT_PATH
                    dict_pred = dict_predict(path2)
                    etiquetas = [dict_pred.get(int(c), 'Desconocido') for c in y_pred]
                    conteo = Counter(etiquetas)
                    normal_count = conteo.get('Normal',0)
                    ddos_tcp_count = conteo.get('DDoS_TCP',0)
                    ddos_udp_count = conteo.get('DDoS_UDP',0)
                    reconnaissance_count = conteo.get('Reconnaissance',0)
                    total = normal_count + ddos_tcp_count + ddos_udp_count + reconnaissance_count
                    labels = ['Normal','DDOS_TCP','DDOS_UDP','Reconnaissance']
                    values = [normal_count, ddos_tcp_count, ddos_udp_count, reconnaissance_count]
                    resumen_text = f"Se analizaron {total} paquetes. Normal: {normal_count}, DDOS_TCP: {ddos_tcp_count}, DDOS_UDP: {ddos_udp_count}, Reconnaissance: {reconnaissance_count}."
                elif model_option=='AUTOENCODER':
                    normal_count = int(np.count_nonzero(y_pred == 0))
                    ddos_tcp_count = int(np.count_nonzero(y_pred == 1))
                    reconnaissance_count = int(np.count_nonzero(y_pred == 2))
                    ddos_udp_count = int(np.count_nonzero(y_pred == 3))
                    total = normal_count + ddos_tcp_count + ddos_udp_count + reconnaissance_count
                    labels = ['Normal','DDOS_TCP','DDOS_UDP','Reconnaissance']
                    values = [normal_count, ddos_tcp_count, ddos_udp_count, reconnaissance_count]
                    resumen_text = f"Se analizaron {total} paquetes. Normal: {normal_count}, DDOS_TCP: {ddos_tcp_count}, DDOS_UDP: {ddos_udp_count}, Reconnaissance: {reconnaissance_count}."
                else:
                    n_normales = int((y_pred == 0).sum())
                    n_anomalos = int((y_pred == 1).sum())
                    total = n_normales + n_anomalos
                    labels = ['Normal','Anómalo']
                    values = [n_normales, n_anomalos]
                    resumen_text = f"Se analizaron {total} paquetes. Normales: {n_normales}, Anómalos: {n_anomalos}."

                # Guardar estado para el panel derecho
                st.session_state['prediccion_realizada'] = True
                st.session_state['resumen_prediccion'] = {'texto': resumen_text, 'labels': labels, 'values': values}
                st.session_state.data = data_df

                # Visual de resultados mínima
                st.markdown("### 📊 Paquetes capturados")
                st.dataframe(data_df)

                # Colores: verde para Normal, rojo para el resto (ataques)
                bar_colors = [('#00C853' if (str(lbl).lower() == 'normal') else '#FF5252') for lbl in labels]
                fig = go.Figure(data=[go.Bar(x=labels, y=values, text=values, textposition='auto', marker_color=bar_colors)])
                fig.update_layout(title='Distribución de Detecciones', title_x=0.5)
                st.plotly_chart(fig, use_container_width=True)

                if metric_option == 'Internas':
                    unique_labels = np.unique(y_pred)
                    if len(unique_labels) >= 2:
                        silhouette, calinski, davies = metricas.metrica_internas(pp3, y_pred)
                        mostrar_metricas(silhouette, calinski, davies)

                # Métricas de recursos
                tiempo_total = time.time() - start_time
                met_cols = st.columns(4)
                with met_cols[0]:
                    st.metric("⏱️ Tiempo (s)", f"{tiempo_total:.2f}")
                with met_cols[1]:
                    st.metric("🖥️ % CPU", f"{cpu_usage:.2f}%")
                with met_cols[2]:
                    st.metric("💾 % Memoria", f"{mem_usage:.2f}%")
                with met_cols[3]:
                    st.metric("📀 % Disco", f"{disco2[3]:.2f}%")

            except Exception as e:
                st.error(f"❌ Error en la predicción automática: {str(e)}")

        # Streamlit UI
        st.title("🔍 Captura y Análisis de Tráfico de Red")

        traffic_method = st.selectbox(
            "📌 Seleccione el método de tráfico",
            ('Iniciar captura de paquetes', 'Abrir captura de paquetes pcap')
        )

        data = None  # Inicializamos el DataFrame

        if traffic_method == 'Iniciar captura de paquetes':

            # Parámetros manuales
            col_man_a, col_man_b = st.columns(2)
            with col_man_a:
                manual_packets = st.number_input('Paquetes a capturar (manual)', min_value=10, max_value=50000, value=100, step=10)
            with col_man_b:
                manual_duration_min = st.number_input('Duración máx. (min) (manual)', min_value=0, max_value=1440, value=0, step=1)
            manual_duration = int(manual_duration_min * 60)
           

            if st.button("🚀 Iniciar Captura"):
                process = capturing_packets(comm_arg)
                st.write("📡 Capturando paquetes...")

                # Inicializamos el DataFrame
                captured_data = None
                progress_bar = st.progress(0, text="Capturando paquetes...")
                total_packets = manual_packets
                start_ts = time.time()

                i = 0
                while i < total_packets:
                    # Parar por tiempo si está configurado
                    if manual_duration and (time.time() - start_ts) >= manual_duration:
                        break
                    packet_str = process.stdout.readline().strip()
                    if packet_str:
                        packet_list = packet_str.split("\t")
                        type_packet_data = type_packet(packet_list)
                        captured_data = packet_df(type_packet_data, captured_data)
                        i += 1
                        progress_bar.progress(i / total_packets, text=f"Paquetes capturados: {i}/{total_packets}")

                try:
                    process.terminate()
                except Exception:
                    pass
                progress_bar.empty()
                st.success("✅ Captura finalizada.")

                # Mostrar el DataFrame capturado y permitir guardar
                if captured_data is not None and list(captured_data.columns) != ['value']:
                    st.write("📊 *Paquetes capturados:*")
                    st.dataframe(captured_data)
                    st.session_state.data = captured_data
                    st.success("Datos capturados listos para predicción")
                    if st.button("💾 Guardar en CSV"):
                        save_packet_csv(captured_data, "captura_paquetes.csv")

            # Ejecución automática si está activa y toca ciclo
            if st.session_state.auto_capture['active'] and not st.session_state.auto_capture['is_capturing']:
                due = st.session_state.auto_capture['next_run_ts'] is not None and time.time() >= st.session_state.auto_capture['next_run_ts']
                if due:
                    st.session_state.auto_capture['is_capturing'] = True
                    st.info('🚀 Ejecutando captura automática...')
                    process = capturing_packets(comm_arg)

                    captured_data = None
                    duration_sec = int(st.session_state.auto_capture['duration_sec'])
                    if duration_sec <= 0:
                        duration_sec = 60
                    start_ts = time.time()
                    end_ts = start_ts + duration_sec
                    prog = st.progress(0, text='Capturando (automática)...')
                    while time.time() < end_ts:
                        if st.session_state.auto_capture['stop_requested']:
                            break
                        packet_str = process.stdout.readline().strip()
                        if packet_str:
                            packet_list = packet_str.split('\t')
                            type_packet_data = type_packet(packet_list)
                            captured_data = packet_df(type_packet_data, captured_data)
                        elapsed = time.time() - start_ts
                        frac = min(1.0, elapsed / duration_sec)
                        prog.progress(frac, text=f"Tiempo transcurrido: {int(elapsed)}s / {duration_sec}s")
                    try:
                        process.terminate()
                    except Exception:
                        pass
                    prog.empty()

                    if st.session_state.auto_capture['stop_requested']:
                        st.warning('⏹️ Captura automática detenida por el usuario.')
                        st.session_state.auto_capture['active'] = False
                        st.session_state.auto_capture['is_capturing'] = False
                        st.session_state.auto_capture['next_run_ts'] = None
                    else:
                        if captured_data is not None and list(captured_data.columns) != ['value']:
                            st.success('✅ Captura automática finalizada. Ejecutando predicción...')
                            st.session_state.data = captured_data
                            run_prediction_on_data(captured_data)
                        # Programar siguiente ciclo y relanzar el countdown
                        st.session_state.auto_capture['next_run_ts'] = time.time() + int(st.session_state.auto_capture['interval_sec'])
                        st.session_state.auto_capture['is_capturing'] = False
                        st.rerun()

        # Sección de carga de archivo CSV
        uploaded_file = None  # Evita UnboundLocalError si no se usa el cargador
        with st.container():
            st.markdown("### 📊 Cargar Datos para Predicción")
            
            # Si hay datos capturados, mostrar mensaje
            if 'data' in st.session_state and st.session_state.data is not None:
                st.info("✅ Usando datos de la captura de paquetes")
                data = st.session_state.data
                st.markdown('<div class="notification" style="background-color: #4CAF50; color: white;">✅ Datos capturados listos para predicción</div>', unsafe_allow_html=True)
            else:
                # De lo contrario, mostrar el cargador de archivo CSV
                uploaded_file = st.file_uploader(
                    "📂 Cargar archivo CSV",
                    type="csv",
                    help="Cargue sus datos de red en formato CSV",
                    key="csv_uploader_1"
                )
                
                if uploaded_file is not None:
                    data = pd.read_csv(uploaded_file)
                    st.markdown('<div class="notification" style="background-color: #4CAF50; color: white;">✅ Datos cargados exitosamente</div>', unsafe_allow_html=True)
                    
                    with st.expander("📊 Vista previa de datos"):
                        st.dataframe(data.head())
                        st.info(f"Dimensiones del dataset: {data.shape[0]} filas, {data.shape[1]} columnas")
                else:
                    data = None

        if st.button('🚀 Realizar Predicción', key='predict'):
            if data is not None:
                run_prediction_on_data(data)
            else:
                st.warning("⚠ Por favor capture paquetes o cargue un archivo CSV antes de realizar la predicción.")

        # Guardar resultados en CSV
        st.markdown("### 💾 Guardar Resultados")
        LABEL = "Ingresa el nombre para tu archivo" 
        
        # 2. Abrimos el div contenedor
        st.markdown('<div class="mi-caja">', unsafe_allow_html=True)

        # 3. Este text_input (label + input) queda dentro de .mi-caja
        nombre_archivo = st.text_input(LABEL)

        # 4. Cerramos el div
        st.markdown('</div>', unsafe_allow_html=True)

        if st.button("Guardar csv"):
            data = st.session_state.data
            y_pred = st.session_state.y_pred_etiqueta

            # Si y_pred es una lista o array, conviértelo en DataFrame para concatenar
            if not isinstance(y_pred, pd.DataFrame):
                y_pred = pd.DataFrame(y_pred, columns=["predicción"])

            resultado = pd.concat([data.reset_index(drop=True), y_pred.reset_index(drop=True)], axis=1)
            resultado.to_csv(nombre_archivo+'.csv', index=False)
            st.success("✅ CSV guardado correctamente.")
        else:
            st.warning("⚠ Primero debes capturar tráfico o hacer una predicción antes de guardar.") 
        
    

    with col2:
        if 'uploaded_file' in locals() and uploaded_file is not None:
            with st.spinner('⏳ Procesando datos...'):
                data = pd.read_csv(uploaded_file)
                st.markdown('<div class="notification" style="background-color: #4CAF50; color: white;">✅ Datos cargados exitosamente</div>', unsafe_allow_html=True)
                time.sleep(3)

                with st.expander("📊 Vista previa de datos"):
                    st.dataframe(data.head())
                    st.info(f"Dimensiones del dataset: {data.shape[0]} filas, {data.shape[1]} columnas")

    # Nota: Bloque antiguo de predicción con archivo subido eliminado por duplicado y uso de modelos no permitidos.
            
            # New Traffic Pattern Visualization
            st.markdown("### 📈 Patrón de Tráfico y Anomalías")
            
            # Calculate traffic metrics
            traffic_data = data.mean(axis=1)  # Using mean of all features as traffic indicator
            
            # Calculate ceiling and floor
            window = 20  # Window size for rolling calculations
            ceiling = traffic_data.rolling(window=window).max()
            floor = traffic_data.rolling(window=window).min()
            
            # Create traffic pattern figure
            fig_traffic = go.Figure()

            # Add main traffic line
            fig_traffic.add_trace(go.Scatter(
                x=list(range(len(traffic_data))),
                y=traffic_data,
                mode='lines',
                name='Tráfico',
                line=dict(color='#4B9FE1', width=1)
            ))

            # Add ceiling line
            fig_traffic.add_trace(go.Scatter(
                x=list(range(len(ceiling))),
                y=ceiling,
                mode='lines',
                name='Techo',
                line=dict(color='#00C853', width=2, dash='dash')
            ))

            # Add floor line
            fig_traffic.add_trace(go.Scatter(
                x=list(range(len(floor))),
                y=floor,
                mode='lines',
                name='Suelo',
                line=dict(color='#FFA726', width=2, dash='dash')
            ))

            # Add anomaly points
            anomaly_indices = np.where(y_pred == 1)[0]
            fig_traffic.add_trace(go.Scatter(
                x=anomaly_indices,
                y=traffic_data[anomaly_indices],
                mode='markers',
                name='Anomalías',
                marker=dict(
                    color='#FF5252',
                    size=8,
                    symbol='circle'
                )
            ))

            # Update layout
            fig_traffic.update_layout(
                title="Patrón de Tráfico y Detección de Anomalías",
                title_x=0.5,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01,
                    bgcolor="rgba(0,0,0,0)"
                ),
                margin=dict(t=50, l=50, r=50, b=50),
                xaxis=dict(
                    title="Muestras",
                    showgrid=False,
                    showline=True,
                    linecolor='rgba(255,255,255,0.2)'
                ),
                yaxis=dict(
                    title="Intensidad de Tráfico",
                    showgrid=True,
                    gridcolor='rgba(255,255,255,0.1)',
                    zeroline=False
                )
            )

            st.plotly_chart(fig_traffic, use_container_width=True)
                
            # Nota: Métricas externas y tabla detallada se gestionan en la página de comparación.


# Control de navegación
def show_compare_page():
    st.title('📊 Comparar Modelos')
    st.markdown(
        """
        <div style="background: rgba(255,255,255,0.06); padding: 1rem 1.2rem; border-radius: 12px; margin: .6rem 0 1rem 0;">
        <p style="margin:0; font-size: .95rem; line-height:1.5;">
        • Seleccione uno o varios modelos para ejecutar la misma predicción sobre un único conjunto de datos (última captura o un CSV).<br/>
        • <strong>Internas</strong>: calcula Silhouette, Calinski y Davies sobre las representaciones del pipeline si hay ≥2 clases predichas.<br/>
        • <strong>Externas</strong>: requiere una columna con etiquetas reales; para modelos multiclase (KMEANS/Autoencoder) se resume a <em>Normal</em> vs <em>Ataque</em> para AUC/ROC.<br/>
        • El gráfico "Normal vs Ataques" suma todas las categorías de ataque (p. ej., DDoS_TCP/UDP, Reconnaissance).
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    src_cols = st.columns(2)
    data_source = None
    if 'data' in st.session_state and st.session_state.data is not None:
        with src_cols[0]:
            usar_captura = st.checkbox('Usar datos de la última captura', value=True)
        if usar_captura:
            data_source = st.session_state.data.copy()
    if data_source is None:
        with src_cols[1]:
            uploaded_cmp = st.file_uploader('Cargar CSV para comparar', type='csv', key='csv_compare')
        if uploaded_cmp is not None:
            data_source = pd.read_csv(uploaded_cmp)
            st.success('Datos cargados para comparar.')
        else:
            st.info('Cargue un CSV o use la última captura para habilitar la comparación.')

    st.markdown('#### Modelos a comparar')
    model_choices = ['IForest', 'OCSVM', 'KMEANS', 'AUTOENCODER']
    selected_models = st.multiselect('Seleccione modelos', model_choices, default=['IForest','OCSVM','KMEANS'])

    metric_compare = st.selectbox('Métricas de validación', ['Internas', 'Externas', 'Ninguna'], help='Internas: Silhouette, Calinski, Davies. Externas: requiere columna de etiquetas reales.')

    true_label_col = None
    if metric_compare == 'Externas':
        if data_source is not None:
            posibles = data_source.columns.tolist()
            true_label_col = st.selectbox('Columna de etiquetas reales (0/1 o multiclase)', posibles)
        else:
            st.info('Cargue datos con etiquetas reales para métricas externas.')

    def _prep_df(df: pd.DataFrame) -> pd.DataFrame:
        lista = ['ip_ttl','tos','ip_flags_rb','ip_flags_df','ip_flags_mf']
        df = df.copy()
        if set(lista).issubset(set(df.columns)):
            df[lista] = df[lista].astype('str')
            df.ip_ttl = df.ip_ttl.str.replace(',', '.')
            df.tos = df.tos.str.replace(',', '.')
            df.ip_flags_rb = df.ip_flags_rb.str_replace(',', '.') if hasattr(df.ip_flags_rb, 'str_replace') else df.ip_flags_rb.str.replace(',', '.')
            df.ip_flags_df = df.ip_flags_df.str.replace(',', '.')
            df.ip_flags_mf = df.ip_flags_mf.str.replace(',', '.')
        return df.replace('', np.nan).fillna(0)

    def _load(opt: str):
               return {
            'IForest': IFOREST_MODEL_PATHS,
            'OCSVM': OCSVM_MODEL_PATH,
            'KMEANS': KMEANS_MODEL_PATHS,
            'AUTOENCODER': AUTOENCODER_MODEL_PATH,
        }.get(opt)

    def _predict(opt: str, model, df: pd.DataFrame):
        try:
            if opt == 'IForest':
                scores = model.decision_function(df)
                y = pred_threshold(scores, -0.10)['Pred'].values
                pp3 = model[0].transform(df)
            # LOF no permitido
            elif opt == 'OCSVM':
                scores = model.decision_function(df)
                thr = np.percentile(scores, 100 * 0.45)
                y = np.where(scores >= thr, 0, 1)
                pp3 = model[0].transform(df)
            elif opt == 'KMEANS':
                y = model.predict(df)
                pp3 = model[0].transform(df)
            # DBSCAN no permitido
            elif opt == 'AUTOENCODER':
                y = model.predict(df)
                pp3 = model[0].transform(df)
            else:
                return None, None
            return pp3, y
        except Exception:
            return None, None

    def _counts(opt: str, y):
        if y is None:
            return {'Normal': 0, 'Anómalo': 0, 'DDOS_TCP': 0, 'DDOS_UDP': 0, 'Reconnaissance': 0}
        if opt in ['KMEANS']:
            path2 = KMEANS_TXT_PATH
            mapping = dict_predict(path2)
            etiquetas = [mapping.get(int(c), 'Desconocido') for c in y]
            c = Counter(etiquetas)
            return {
                'Normal': int(c.get('Normal', 0)),
                'Anómalo': int(c.get('Anómalo', 0)),
                'DDOS_TCP': int(c.get('DDoS_TCP', 0)),
                'DDOS_UDP': int(c.get('DDoS_UDP', 0)),
                'Reconnaissance': int(c.get('Reconnaissance', 0)),
            }
        if opt == 'AUTOENCODER':
            return {
                'Normal': int(np.count_nonzero(y == 0)),
                'Anómalo': 0,
                'DDOS_TCP': int(np.count_nonzero(y == 1)),
                'DDOS_UDP': int(np.count_nonzero(y == 3)),
                'Reconnaissance': int(np.count_nonzero(y == 2)),
            }
        normales = int((y == 0).sum())
        anomalos = int((y == 1).sum())
        return {'Normal': normales, 'Anómalo': anomalos, 'DDOS_TCP': 0, 'DDOS_UDP': 0, 'Reconnaissance': 0}

    if st.button('🚀 Comparar'):
        if data_source is None:
            st.warning('⚠ Cargue datos o use la última captura antes de comparar.')
            return
        dfp = _prep_df(data_source)
        y_true = None
        if metric_compare == 'Externas' and true_label_col is not None and true_label_col in data_source.columns:
            y_true = data_source[true_label_col].values
        filas = []
        for opt in selected_models:
            path = _load(opt)
            if path is None:
                st.error(f'Ruta no definida para {opt}')
                continue
            model = load_model(path)
            if model is None:
                st.error(f'No se pudo cargar el modelo {opt}')
                continue
            t0 = time.time()
            pp3, y = _predict(opt, model, dfp)
            pred_time = time.time() - t0
            conteos = _counts(opt, y)
            fila = {'Modelo': opt, **conteos, 'Tiempo (s)': float(pred_time)}
            if metric_compare == 'Internas' and y is not None and len(np.unique(y)) >= 2:
                try:
                    s, c, d = metricas.metrica_internas(pd.DataFrame(pp3), y)
                    fila.update({'Silhouette': float(s), 'Calinski': float(c), 'Davies': float(d)})
                except Exception:
                    fila.update({'Silhouette': None, 'Calinski': None, 'Davies': None})
            if metric_compare == 'Externas' and y_true is not None and y is not None and len(y_true) == len(y):
                try:
                    from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
                    # Si y es multiclase, intentamos binarizar para ROC (Normal=0 vs resto=1)
                    y_bin = None
                    if opt in ['KMEANS']:
                        # mapear etiquetas a binario: Normal=0, otros=1
                        path2 = KMEANS_TXT_PATH
                        mapping = dict_predict(path2)
                        etiquetas = np.array([mapping.get(int(c), 'Desconocido') for c in y])
                        y_bin = (etiquetas != 'Normal').astype(int)
                    elif opt == 'AUTOENCODER':
                        y_bin = (y != 0).astype(int)
                    else:
                        y_bin = y.astype(int)

                    # confusión
                    cm = confusion_matrix(y_true, y_bin)
                    fila.update({'MatrizConfusion': cm})

                    # ROC AUC si y_true es binario
                    if np.unique(y_true).size == 2 and np.unique(y_bin).size == 2:
                        try:
                            aucv = roc_auc_score(y_true, y_bin)
                            fila.update({'ROC_AUC': float(aucv)})
                        except Exception:
                            pass
                except Exception:
                    pass
            filas.append(fila)

        if not filas:
            st.info('Sin resultados para mostrar.')
            return

        res_df = pd.DataFrame(filas)
        cols = ['Modelo', 'Normal', 'Anómalo', 'DDOS_TCP', 'DDOS_UDP', 'Reconnaissance', 'Tiempo (s)']
        if metric_compare == 'Internas':
            cols += ['Silhouette', 'Calinski', 'Davies']
        if metric_compare == 'Externas':
            cols += ['ROC_AUC']
        st.markdown('### 📋 Resumen')
        st.dataframe(res_df[cols], use_container_width=True)

        if metric_compare == 'Externas':
            st.markdown('### 🧮 Matrices de Confusión')
            for _, row in res_df.iterrows():
                if 'MatrizConfusion' in row and row['MatrizConfusion'] is not None:
                    st.write(f"Modelo: {row['Modelo']}")
                    st.dataframe(pd.DataFrame(row['MatrizConfusion']), use_container_width=True)

        st.markdown('### 📊 Normal vs Ataques')
        modelos = res_df['Modelo']
        normales = res_df['Normal']
        ataques = (res_df['Anómalo'].fillna(0) + res_df['DDOS_TCP'].fillna(0) + res_df['DDOS_UDP'].fillna(0) + res_df['Reconnaissance'].fillna(0))
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Normal', x=modelos, y=normales, marker_color='#00C853'))
        fig.add_trace(go.Bar(name='Ataques', x=modelos, y=ataques, marker_color='#FF5252'))
        fig.update_layout(barmode='group', title='Normal vs Ataques por modelo', title_x=0.5)
        st.plotly_chart(fig, use_container_width=True)

        # Gráfico ligero de tiempos
        if 'Tiempo (s)' in res_df.columns:
            st.markdown('### ⏱️ Tiempos de predicción por modelo')
            figt = go.Figure(data=[go.Bar(x=res_df['Modelo'], y=res_df['Tiempo (s)'], marker_color='#4B9FE1')])
            figt.update_layout(title='Tiempo de predicción (s)', title_x=0.5)
            st.plotly_chart(figt, use_container_width=True)

if st.session_state.page == "home":
    show_home_page()
elif st.session_state.page == "dashboard":
    show_dashboard_page()
else:
    show_compare_page()

# Pie de página mejorado
st.markdown("---")
col1, col2, col3 = st.columns([1,2,1])
with col2:
    st.markdown(
        """
        <div style='text-align: center'>
            <p>🛡 Sistema de Detección de Anomalías IoT</p>
            <p>© 2025 Innovasic | UCC</p>
        </div>
        """,
        unsafe_allow_html=True)
