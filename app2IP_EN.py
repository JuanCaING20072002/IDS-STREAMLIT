import streamlit as st
from autoencoder_classifier import AutoencoderClassifier
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
from i18n import t, set_language, get_language


# Global load_model message translations
def load_model(model_path):
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.error(f"Error details: {traceback.format_exc()}")
        return None

def dict_predict(path2):
    # Lee todo el contenido
    with open(path2, 'r') as f:
        raw = f.read()

    # Convierte la cadena a la lista de tuplas
    lista_tuplas = ast.literal_eval(raw)  # -> [(0, 'Normal'), (1, 'DDoS_TCP'), …]

    # Construye el dict de un paso
    dict_predict = dict(lista_tuplas)
    return dict_predict

def pred_threshold(
    score, threshold):
    score = pd.Series(score)  # ✅ Convertir a Series si es un array
    Actual_pred = pd.DataFrame({'Pred': score})  
    Actual_pred['Pred'] = np.where(Actual_pred['Pred']<=threshold,0,1)
    Actual_pred=Actual_pred.reset_index(drop=True)
    return (Actual_pred)

def dbscan_predict(dbscan, X_new):
    """
    Asigna cada punto de X_new al clúster del vecino core más cercano
    usando el DBSCAN ya entrenado.
    """
    # 1) Extrae los core samples y sus etiquetas
    core_samples = dbscan.components_
    core_labels  = dbscan.labels_[dbscan.core_sample_indices_]

    # 2) Ajusta un NearestNeighbors sobre esos core samples
    nn = NearestNeighbors(n_neighbors=1).fit(core_samples)

    # 3) Para cada punto nuevo, obtiene distancia y índice de vecino
    dist, idx = nn.kneighbors(X_new)

    # 4) Si la distancia <= eps, asigna la etiqueta de ese vecino; si no, -1
    return np.where(dist.ravel() <= dbscan.eps,
                    core_labels[idx.ravel()],
                    -1)

# IMPORTANTE: st.set_page_config debe ser la primera llamada a Streamlit en el script
st.set_page_config(
    page_title=t("app.title"),
    page_icon="🛡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Selector de idioma en la barra lateral
lang = st.sidebar.selectbox(
    label=t("app.language"),
    options=["es", "en"],
    index=["es", "en"].index(get_language()) if get_language() in ["es","en"] else 1,
    help="UI language"
)
set_language(lang)

# Navegación mejorada
with st.sidebar:
    st.image("https://img.icons8.com/color/48/000000/shield.png", width=50)
    selected = option_menu(
        t("nav.title"),
        [t("app.home"), t("app.dashboard")],
        icons=["house", "graph-up"],
        menu_icon="cast",
        default_index=0
    )
    st.session_state.page = "home" if selected == t("app.home") else "dashboard"

# Función para cargar animaciones Lottie
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        st.error(f"Error loading animation from {url}. HTTP status: {r.status_code}")
        return None
    return r.json()

# Animaciones
#lottie_loading = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_szlepvdh.json")
#lottie_how_it_works = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_yd8fbnml.json")
#lottie_network = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_ggwq3ysg.json") 
#lottie_success = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_ktwnwv5m.json")  
#lottie_security = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_jcsfwbvi.json")  

# Verificar que las animaciones se han cargado correctamente
#if lottie_loading is None or lottie_success is None:
    #st.error("Error al cargar las animaciones Lottie. Por favor, verifica las URLs.")

# Inicializar el estado de la sesión para la navegación
#if "page" not in st.session_state:
    #st.session_state.page = "home"

#----------------------------------------------------  Estilos CSS --------------------------------------------------------------------------------------------------------------------------------
st.markdown("""
<style>
    body {
        color: #ffffff;
        font-family: 'Roboto', sans-serif;
    }
    .mi-caja label {
        color: black !important;         /* etiqueta en negro */
    }
    .mi-caja input {
        color: black !important;         /* texto que se escribe en negro */
        background-color: white !important;
        caret-color: black !important;
    }

        .main-header {
        font-family: 'Roboto', sans-serif;
        background: linear-gradient(90deg, #1E3D59 0%, #2E5077 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 30px;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.1);
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 10px 0;
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    .success-animation {
        animation: fadeInOut 3s forwards;
    }
    @keyframes fadeInOut {
        0% { opacity: 0; }
        20% { opacity: 1; }
        80% { opacity: 1; }
        100% { opacity: 0; display: none; }
    }
    .stAlert {
        background-color: rgba(25, 25, 25, 0.5);
        color: white;
        border: none;
        padding: 1rem;
        border-radius: 10px;
    }
    .stButton > button {
        background: rgba(255, 255, 255, 0.1);
        border: none;
        color: #ffffff;
        padding: 10px 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1), 0 1px 3px rgba(0, 0, 0, 0.08);
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background: rgba(255, 255, 255, 0.2);
        transform: translateY(-2px);
        box-shadow: 0 7px 14px rgba(0, 0, 0, 0.1), 0 3px 6px rgba(0, 0, 0, 0.08);
    }
    .stButton > button:active {
        transform: translateY(1px);
        box-shadow: 0 3px 8px rgba(0, 0, 0, 0.1), 0 1px 3px rgba(0, 0, 0, 0.08);
    }
    .dashboard-form {
        background: rgba(255, 255, 255, 0.1);
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        max-width: 600px;
        margin: auto;
    }
    .notification {
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
        animation: fadeOut 2s forwards;
        animation-delay: 3.5s;
    }
    @keyframes fadeOut {
        from {opacity: 1;}
        to {opacity: 0; height: 0; padding: 0; margin: 0;}
    }
    .streamlit-expanderHeader, .stTextInput > div > div > input {
        color: #ffffff !important;
        background-color: rgba(255, 255, 255, 0.1) !important;
    }
    .stDataFrame {
        color: #ffffff;
    }
    .card {
        background: rgba(255, 255, 255, 0.1);
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .carousel {
        display: flex;
        overflow-x: auto;
        scroll-snap-type: x mandatory;
    }
    .carousel-item {
        flex: none;
        scroll-snap-align: start;
        margin-right: 20px;
    }
    
""", unsafe_allow_html=True)

# ---------------------------------------------------- INICIO PAGE----------------------------------------------------------------------------------------------------------------------------

def show_home_page():
    st.title("🛡 " + t("home.title"))

    st.write(t("home.intro"))

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🎯 " + t("home.objectives.title"))
        st.markdown(t("home.objectives.items"))

    with col2:
        pass

    st.markdown("---")
    st.subheader("💬 " + t("home.results.title"))

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(t("home.results.validation"))

    with col2:
        st.markdown(t("home.results.classes"))

    with col3:
        st.markdown(t("home.results.compute"))

    st.markdown("---")
    st.subheader("🔎 " + t("home.models.title"))
    with st.expander(t("home.models.ocsvm.title")):
        st.write(t("home.models.ocsvm.desc"))
    with st.expander(t("home.models.iforest.title")):
        st.write(t("home.models.iforest.desc"))
    with st.expander(t("home.models.kmeans.title")):
        st.write(t("home.models.kmeans.desc"))
    with st.expander(t("home.models.autoencoder.title")):
        st.write(t("home.models.autoencoder.desc"))

    st.markdown("---")
    st.subheader("📖 " + t("home.howto.title"))
    st.write(t("home.howto.lead"))
    st.markdown(
        f"""
        <div class="carousel">
            <div class="carousel-item">
                <h4>{t('home.howto.step1.title')}</h4>
                <p>{t('home.howto.step1.desc')}</p>
            </div>
            <div class="carousel-item">
                <h4>{t('home.howto.step2.title')}</h4>
                <p>{t('home.howto.step2.desc')}</p>
            </div>
            <div class="carousel-item">
                <h4>{t('home.howto.step3.title')}</h4>
                <p>{t('home.howto.step3.desc')}</p>
            </div>
            <div class="carousel-item">
                <h4>{t('home.howto.step4.title')}</h4>
                <p>{t('home.howto.step4.desc')}</p>
            </div>
            <div class="carousel-item">
                <h4>{t('home.howto.step5.title')}</h4>
                <p>{t('home.howto.step5.desc')}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.subheader("📊 " + t("home.reco.title"))
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f"""
            <div class="card">
                <h4>✅ {t('home.reco.normal.title')}</h4>
                <p>{t('home.reco.normal.desc')}</p>
                <p><strong>{t('home.reco.recommendation')}:</strong> {t('home.reco.normal.action')}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class="card">
                <h4>⚠ {t('home.reco.anomalous.title')}</h4>
                <p>{t('home.reco.anomalous.desc')}</p>
                <p><strong>{t('home.reco.recommendation')}:</strong> {t('home.reco.anomalous.action')}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""
            <div class="card">
                <h4>⚠ {t('home.reco.attacks.title')}</h4>
                {t('home.reco.attacks.content')}
            </div>
            """,
            unsafe_allow_html=True,
        )

#---------------------------------------------------- DASHBOARD ----------------------------------------------------------------------------------------------------------------------------
# DistanceMetric.get_metric("euclidean") devuelve un objeto cuya clase interna
# es la que Python debe encontrar al desempaquetar el pickle.
_dm.EuclideanDistance = DistanceMetric.get_metric("euclidean").__class__



def show_dashboard_page():
    st.title('❇ ' + t('dashboard.title'))

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
    
    
    
    # Rutas a los modelos
         
    LOF_MODEL_PATHS = "model/lof__EJERCICIO10_n_components6_fit_over.pkl"
    IFOREST_MODEL_PATHS = "model/iforest__EJERCICIO10_n_components6_fit.pkl"
    KMEANS_MODEL_PATHS = "model/kmeans1__EJERCICIO10_n_components6_over.pkl"
    DBSCAN_MODEL_PATHS= "model/dbscan__EJERCICIO10_n_components6_fit_over.pkl"
    DBSCAN_TXT_PATH="model/list_dbscan2_over.txt"
    KMEANS_TXT_PATH="model/list_kmeans_over.txt"
    AUTOENCODER_MODEL_PATH="model/autoencoder.pkl"
    OCSVM_MODEL_PATH="model/ocsvm__EJERCICIO10_n_components6_fit.pkl"


    def load_model(model_path):
        try:
            absolute_path = os.path.abspath(model_path)
            st.write(f"Attempting to load from: {absolute_path}")
            if not os.path.exists(absolute_path):
                raise FileNotFoundError(f"File not found: {absolute_path}")
            model = joblib.load(absolute_path)
            st.write(f"Loaded model type: {type(model)}")
            return model
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            st.error(f"Error details: {traceback.format_exc()}")
            return None

    col1, col2 = st.columns([1, 2])
    
    # Interfaz Usuario Panel de Control
    with col1:
        st.markdown("### 🖥 " + t("dashboard.control_panel"))
        model_option = st.selectbox(
            "📌 " + t("dashboard.model_select"),
            ('IForest','OCSVM','KMEANS','AUTOENCODER'),
            help=t("dashboard.model_help")
        )
        metric_choice = st.selectbox(
            "📌 " + t("dashboard.metric_select"),
            ['external', 'internal'],
            format_func=lambda v: t('dashboard.metric.' + v),
            help=t("dashboard.metric_help")
        )
        # normalizamos a la variable existente si más abajo se usa metric_option
        metric_option = metric_choice

        # Función para procesar los paquetes capturados
        # ----------------------------------------------
        # Configuración del comando para Tshark
        comm_arg =  (
    "sudo /usr/local/bin/tshark "
    "-i eth0 -i wlan0 "
    "-l -c 1000 "
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
            if df is None:
                df = pd.DataFrame(type_packet)
            else:
                df = pd.concat([df, pd.DataFrame(type_packet)], axis=0, ignore_index=True)
                
                
            return df
        
        def predecir(model, data, model_option):
            try:
                
                
                # Obtener las columnas desde el preprocesador
                columns = model[0].named_steps['prepro_2_del']\
                  .named_steps['prepro_1_num_cat']\
                  .get_feature_names_out().tolist()

                # Reemplazar del_columns en el pipeline con la versión correcta
                model[0].named_steps['prepro_2_del'].named_steps['del_columns'] = del_columns(columns)

              
                
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
                return None, None
        
        # Update metrics cards text
        def mostrar_metricas(silhouette, calinski, davies):
            st.markdown("### 📊 Internal Metrics")
            cols = st.columns(3)
            with cols[0]:
                st.metric("Silhouette Score", f"{silhouette:.3f}", delta="Good" if silhouette > 0.5 else "Fair")
            with cols[1]:
                st.metric("Calinski Score", f"{calinski:.3f}", delta="Good" if calinski > 1000 else "Fair")
            with cols[2]:
                st.metric("Davies Score", f"{davies:.3f}", delta="Good" if davies < 0.5 else "Fair")     
                            
                     

        # Función para guardar los paquetes en CSV
        def save_packet_csv(df, path):
            df.to_csv(path, index=False)
            st.success(f"✅ File saved to {path}")
            
        # Streamlit UI
        st.title("🔍 " + t("dashboard.capture.header"))

        traffic_choice = st.selectbox(
            "📌 " + t("dashboard.traffic_select"),
            ['start_capture', 'open_pcap'] ,
            format_func=lambda v: t('dashboard.traffic.' + v)
        )

        traffic_method = 'Iniciar captura de paquetes' if traffic_choice == 'start_capture' else 'Open pcap capture'

        data = None  # Inicializamos el DataFrame

        if traffic_choice == 'start_capture':
            if st.button("🚀 " + t("dashboard.buttons.start_capture")):
                process = capturing_packets(comm_arg)
                st.write("📡 " + t("dashboard.messages.capturing"))

                # Inicializamos el DataFrame
                captured_data = None

                for _ in range(100):  # Capturar 100 paquetes
                    packet_str = process.stdout.readline().strip()
                    if packet_str:
                        packet_list = packet_str.split("\t")  # Separar los datos por tabulación
                        type_packet_data = type_packet(packet_list)  # Convertir a estructura correcta
                        captured_data = packet_df(type_packet_data, captured_data)  # Agregar al DataFrame

                process.terminate()
                st.success("✅ " + t("dashboard.messages.capture_done"))

                # Show captured DataFrame
                if captured_data is not None:
                    st.write("📊 " + t("dashboard.messages.captured_packets"))
                    st.dataframe(captured_data)
                    
                    # Guardar los datos capturados en la sesión
                    st.session_state.data = captured_data
                    
                    # Mostrar mensaje de confirmación
                    st.success(t("dashboard.messages.data_ready"))
                    if st.button("💾 " + t("dashboard.buttons.save_csv")):
                        save_packet_csv(captured_data, "packet_capture.csv")

        # Sección de carga de archivo CSV
        # Aseguramos variable definida para evitar UnboundLocalError
        uploaded_file = None
        with st.container():
            st.markdown("### 📊 " + t("dashboard.load_data"))
            
            # Si hay datos capturados, mostrar mensaje
            if 'data' in st.session_state and st.session_state.data is not None:
                st.info("✅ Using captured packet data")
                data = st.session_state.data
                st.markdown('<div class="notification" style="background-color: #4CAF50; color: white;">✅ ' + t("dashboard.messages.data_loaded") + '</div>', unsafe_allow_html=True)
            else:
                uploaded_file = st.file_uploader(
                    "📂 " + t("app.upload_csv"),
                    type="csv",
                    help=t("dashboard.upload_help"),
                    key="csv_uploader_1",
                )
                
                if uploaded_file is not None:
                    data = pd.read_csv(uploaded_file)
                    st.markdown('<div class="notification" style="background-color: #4CAF50; color: white;">✅ ' + t("dashboard.messages.data_loaded") + '</div>', unsafe_allow_html=True)
                    
                    with st.expander("📊 " + t("dashboard.preview")):
                        st.dataframe(data.head())
                        st.info(t("dashboard.dataset_dims").format(rows=data.shape[0], cols=data.shape[1]))
                else:
                    data = None

        if st.button('🚀 ' + t('dashboard.buttons.run_prediction'), key='predict'):
            if data is not None:
                loading_placeholder = st.empty()
                #with loading_placeholder.container():
                    #if lottie_loading is not None:
                        #st_lottie(lottie_loading, height=200, key="loading")
                
                try:
                    # Inicializar el modelo
                    model = None
                    
                    # Cargar el modelo según la opción seleccionada
                    if model_option == 'IForest':
                        model = load_model(IFOREST_MODEL_PATHS)
                    elif model_option=='LOF':
                        model=load_model(LOF_MODEL_PATHS)
                    elif model_option=='OCSVM':
                        model=load_model(OCSVM_MODEL_PATH)
                    elif model_option=='KMEANS':
                        model = load_model(KMEANS_MODEL_PATHS)
                    elif model_option=='DBSCAN':
                        model=load_model(DBSCAN_MODEL_PATHS)
                    elif model_option=='AUTOENCODER':
                        model=load_model(AUTOENCODER_MODEL_PATH)
                        if model is None:
                            raise Exception("No se pudo cargar el modelo IForest")
                    if model is None:
                        raise Exception("No se pudo cargar el modelo seleccionado")
                
                    
                    if traffic_method == 'Iniciar captura de paquetes':
                                  
                        lista=['ip_ttl',  'tos','ip_flags_rb','ip_flags_df','ip_flags_mf']
                        #tipo string, para reemplazar un coma por el punto y convertirlo en float
                        data[lista] = data[lista].astype('str')
                        data.ip_ttl = data.ip_ttl.str.replace(',','.')
                        data.tos = data.tos.str.replace(',','.')
                        data.ip_flags_rb = data.ip_flags_rb.str.replace(',','.')
                        data.ip_flags_df = data.ip_flags_df.str.replace(',','.')
                        data.ip_flags_mf = data.ip_flags_mf.str.replace(',','.')
                    #   data[lista] = data[lista].astype('float64')
                        
                        data = data.replace('', np.nan).fillna(0)
                        # Realizar predicción
                        start_time = time.time()
                        psutil.cpu_times()
                        psutil.virtual_memory()
                        psutil.disk_partitions()

                        t_cpu = psutil.cpu_times()
                        start_user = t_cpu.user
                        cpu_usage = psutil.cpu_percent(1)
                        mem_usage = psutil.virtual_memory().percent
                        disco2 = psutil.disk_usage('/')
                        pp3, y_pred,scores = predecir(model, data, model_option)

                        #st.session_state.y_pred = y_pred
                        
                        if(model_option=='KMEANS' or model_option=='DBSCAN'):
                            
                            if(model_option=='KMEANS'):
                                path2=KMEANS_TXT_PATH
                            else:
                                path2=DBSCAN_TXT_PATH
                                
                            dict_pred=dict_predict(path2)
                            #etiqueta=dict_pred[cluster[-1]]
                            etiquetas = [ dict_pred[c] for c in y_pred ]

                            y_pred_etiqueta=etiquetas

                                    
                            # 2) Cuenta cada etiqueta
                            conteo = Counter(etiquetas)

                            # 3) Guarda en un diccionario todas las cuentas
                            label_counts = dict(conteo)
                            normal_count         = label_counts.get('Normal', 0)
                            ddos_tcp_count       = label_counts.get('DDoS_TCP', 0)
                            ddos_udp_count       = label_counts.get('DDoS_UDP', 0)
                            reconnaissance_count = label_counts.get('Reconnaissance', 0)

                            

                        elif(model_option=='AUTOENCODER'):
                            # Supongamos que tus etiquetas predichas están en YA_pred
                            normal_count = np.count_nonzero(y_pred == 0)
                            ddos_tcp_count = np.count_nonzero(y_pred == 1)
                            reconnaissance_count = np.count_nonzero(y_pred == 2)
                            ddos_udp_count = np.count_nonzero(y_pred == 3)

                            

                            # Condiciones y etiquetas
                            condiciones = [
                                y_pred == 0,
                                y_pred == 1,
                                y_pred == 2,
                                y_pred == 3
                            ]
                            etiquetas = [
                                'Normal',
                                'DDoS_TCP',
                                'Reconnaissance',
                                'DDoS_UDP'
                            ]

                            # Reemplazo
                            y_pred_etiqueta = np.select(condiciones, etiquetas, default='Desconocido')

                        else:
                            n_normales  = int((y_pred == 0).sum())
                            n_anomalos  = int((y_pred == 1).sum())

                            # Condiciones y etiquetas
                            condiciones = [
                                y_pred == 0,
                                y_pred == 1,
                            ]
                            etiquetas = [
                                'Normal',
                                'Ataque'
                            ]

                            # Reemplazo
                            y_pred_etiqueta = np.select(condiciones, etiquetas, default='Desconocido')

                        
                            st.write(n_normales,n_anomalos)
                            st.write(scores)

                        st.session_state.y_pred_etiqueta = y_pred_etiqueta
                        
                        
                        
                        
                        if pp3 is None or y_pred is None:
                            raise Exception("Error en la predicción")
                    
                    # --- Parte métricas internas ---
                    # Para cualquier algoritmo necesitamos al menos 2 clusters distintos
                    unique_labels = np.unique(y_pred)
                    if len(unique_labels) >= 2:
                        silhouette, calinski, davies = metricas.metrica_internas(pp3, y_pred)
                        if metric_option == 'internal':
                            mostrar_metricas(silhouette, calinski, davies)
                    else:
                        st.write("Metrics cannot be computed because there is only one cluster")
                    
                    
                    # Limpiar animación de carga
                    time.sleep(3)
                    loading_placeholder.empty()
                    
                    # Sección de visualización
                    # Sección de visualización
                    with col1:
                        # Contar normales y anómalos usando y_pred
                        if(model_option=='IForest' or model_option=='LOF' or model_option=='OCSVM'):
                            st.write(f"Normales: {n_normales}, Anómalos: {n_anomalos}")

                            # Crear gráfico de barras con Plotly
                            fig = go.Figure(data=[
                                go.Bar(
                                    x=['Normal', 'Anómalo'],
                                    y=[n_normales, n_anomalos],
                                    marker_color=['#00C853', '#FF5252'],
                                    text=[f"{n_normales:,}", f"{n_anomalos:,}"],
                                    textposition='auto',
                                    hoverinfo='y+text',
                                    width=0.8
                                )
                            ])
                            fig.update_layout(
                                title="Distribución de Detecciones",
                                title_x=0.5,
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)',
                                font=dict(color='white'),
                                showlegend=False,
                                margin=dict(t=50, l=50, r=50, b=50),
                                width=800,
                                height=700,
                                xaxis=dict(
                                    title="Class",
                                    showgrid=False,
                                    showline=True,
                                    linecolor='rgba(255,255,255,0.2)'
                                ),
                                yaxis=dict(
                                    title="Count",
                                    showgrid=True,
                                    gridcolor='rgba(255,255,255,0.1)',
                                    zeroline=False
                                ),
                                bargap=0.2
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                        elif(model_option=='KMEANS' or model_option=='DBSCAN' or model_option=='AUTOENCODER'):
                            st.write(f"Normal: {normal_count}, DDOS_TCP: {ddos_tcp_count}, DDOS_UDP:{ddos_udp_count}, Reconnaisance:{reconnaissance_count}")

                            # Crear gráfico de barras con Plotly
                            fig = go.Figure(data=[
                                go.Bar(
                                    x=['Normal', 'DDOS_TCP','DDOS_UDP','Reconnaisance'],
                                    y=[normal_count, ddos_tcp_count,ddos_udp_count,reconnaissance_count],
                                    marker_color=['#00C853', '#FF5252', '#FF5252', '#FF5252'],
                                    text=[f"{normal_count:,}", f"{ddos_tcp_count:,}", f"{ddos_udp_count:,}", f"{reconnaissance_count:,}"],
                                    textposition='auto',
                                    hoverinfo='y+text',
                                    width=0.8
                                )
                            ])
                            fig.update_layout(
                                title="Distribución de Detecciones",
                                title_x=0.5,
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)',
                                font=dict(color='white'),
                                showlegend=False,
                                margin=dict(t=50, l=50, r=50, b=50),
                                width=950,
                                height=700,
                                xaxis=dict(
                                    title="Class",
                                    showgrid=False,
                                    showline=True,
                                    linecolor='rgba(255,255,255,0.2)'
                                ),
                                yaxis=dict(
                                    title="Count",
                                    showgrid=True,
                                    gridcolor='rgba(255,255,255,0.1)',
                                    zeroline=False
                                ),
                                bargap=0.2
                            )
                            st.plotly_chart(fig, use_container_width=True)
                
                        st.write('Rendimiento de Gastos Generales:')
                        st.write(f"tiempo:---  { time.time() - start_time} Segundos ---")
                        st.write('tiempo_cpu', t_cpu)
                        st.write('%_cpu', '%.2f%%' % cpu_usage)
                        st.write('%_mem', '%.2f%%' % mem_usage)
                        st.write ('consumo de disco', '%.2f%%' % disco2[3])
                
                
                


                except Exception as e:
                    st.error(f"❌ Processing error: {str(e)}")
                    import traceback
                    st.error(f"Details: {traceback.format_exc()}")
                    loading_placeholder.empty()
            else:
                st.warning("⚠ " + t("dashboard.messages.need_data"))

        LABEL = t("dashboard.filename_label")
        
        # 2. Abrimos el div contenedor
        st.markdown('<div class="mi-caja">', unsafe_allow_html=True)

        # 3. Este text_input (label + input) queda dentro de .mi-caja
        nombre_archivo = st.text_input(LABEL)

        # 4. Cerramos el div
        st.markdown('</div>', unsafe_allow_html=True)

        if st.button(t("dashboard.buttons.save_csv")):
            data = st.session_state.data
            y_pred = st.session_state.y_pred_etiqueta

            # Si y_pred es una lista o array, conviértelo en DataFrame para concatenar
            if not isinstance(y_pred, pd.DataFrame):
                y_pred = pd.DataFrame(y_pred, columns=["prediction"])  # translated header

            resultado = pd.concat([data.reset_index(drop=True), y_pred.reset_index(drop=True)], axis=1)
            resultado.to_csv(nombre_archivo + '.csv', index=False)
            st.success("✅ " + t("dashboard.messages.csv_saved"))
        else:
            st.warning("⚠ " + t("dashboard.messages.save_before"))

    with col2:
        if uploaded_file is not None:
            with st.spinner('⏳ ' + t("dashboard.messages.processing")):
                data = pd.read_csv(uploaded_file)
                st.markdown('<div class="notification" style="background-color: #4CAF50; color: white;">✅ ' + t("dashboard.messages.data_loaded") + '</div>', unsafe_allow_html=True)
                time.sleep(3)

                with st.expander("📊 " + t("dashboard.preview")):
                    st.dataframe(data.head())
                    st.info(t("dashboard.dataset_dims").format(rows=data.shape[0], cols=data.shape[1]))
                
                
            # Matriz de Confusión
            
            def confusion_matrix_threshold(actual,score, threshold):
                Actual_pred = pd.DataFrame({'Actual': actual, 'Pred': score})
                Actual_pred['Pred'] = np.where(Actual_pred['Pred']<=threshold,0,1)
                cm = pd.crosstab(Actual_pred['Actual'],Actual_pred['Pred'])
                return(cm)
            
            if(metric_option=='external'):
                with st.expander("📖 " + t("metrics.explanation")):
                        
                        st.write(t("metrics.external"))
                        
                st.markdown("### 📊 " + t("metrics.confusion_matrix"))
                data_y=datay.values
                matrix_confusion = confusion_matrix_threshold(datay, average_scores, threshold)
                st.table(matrix_confusion.style.format("{:,.0f}"))
                # Explicación interactiva
                with st.expander("📖 ¿Qué es la Matriz de Confusión?"):
                    st.markdown("""
                        La Matriz de Confusión es una herramienta para evaluar el desempeño de un modelo de clasificación. 
                        Muestra la cantidad de predicciones correctas e incorrectas de un modelo en una tabla con cuatro categorías:
                        - Verdaderos positivos (TP): Predicciones correctas de la clase positiva.
                        - Falsos negativos (FN): Predicciones incorrectas donde el modelo predijo la clase negativa.
                        - Verdaderos negativos (TN): Predicciones correctas de la clase negativa.
                        - Falsos positivos (FP): Predicciones incorrectas donde el modelo predijo la clase positiva.
                    """)
                
            if(metric_option=='internal'):
                with st.expander("📖 " + t("metrics.internas.explanation")):
                        
                        st.write(t("metrics.internas.desc"))
                        st.write("""
                        1. Silhouette Score:
                        - El puntaje de silueta es una medida utilizada para evaluar la calidad de un agrupamiento (clustering) en un conjunto de datos.
                        - Se basa en la distancia entre los puntos de datos y su relación con otros grupos.
                        - Un valor cercano a 1 indica que el punto está bien agrupado.
                        - Un valor cercano a 0 indica que el punto está en la frontera entre dos grupos.
                        - Un valor negativo indica que el punto podría estar mal agrupado.

                        2. Calinski-Harabsz Score:

                        3. Davies-bouldin Score:
                        - Se enfoca en la similitud entre pares de clusters, considerando tanto la compacidad como la separación.
                        - Cuanto más bajo sea el índice, más separados y compactos serán los clusters, lo que indica un agrupamiento de mayor calidad.
                        

                        """)    
            
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
                name='Traffic',
                line=dict(color='#4B9FE1', width=1)
            ))

            # Add ceiling line
            fig_traffic.add_trace(go.Scatter(
                x=list(range(len(ceiling))),
                y=ceiling,
                mode='lines',
                name='Ceiling',
                line=dict(color='#00C853', width=2, dash='dash')
            ))

            # Add floor line
            fig_traffic.add_trace(go.Scatter(
                x=list(range(len(floor))),
                y=floor,
                mode='lines',
                name='Floor',
                line=dict(color='#FFA726', width=2, dash='dash')
            ))

            # Add anomaly points
            anomaly_indices = np.where(y_pred == 1)[0]
            fig_traffic.add_trace(go.Scatter(
                x=anomaly_indices,
                y=traffic_data[anomaly_indices],
                mode='markers',
                name='Anomalies',
                marker=dict(
                    color='#FF5252',
                    size=8,
                    symbol='circle'
                )
            ))

            # Update layout
            fig_traffic.update_layout(
                title="Traffic Pattern and Anomaly Detection",
                title_x=0.5,
                xaxis=dict(
                    title="Samples",
                    showgrid=False,
                    showline=True,
                    linecolor='rgba(255,255,255,0.2)'
                ),
                yaxis=dict(
                    title="Traffic Intensity",
                    showgrid=True,
                    gridcolor='rgba(255,255,255,0.1)',
                    zeroline=False
                ),
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
            )

            st.plotly_chart(fig_traffic, use_container_width=True)
                
            with col2:
                
                # Métricas de rendimiento
                X_test_norm = data.values
                if metric_option== 'external':
                    data_y=datay.values
                    #Matriz de confusion
                    def confusion_matrix_threshold(actual,score, threshold):
                        Actual_pred = pd.DataFrame({'Actual': actual, 'Pred': score})
                        Actual_pred['Pred'] = np.where(Actual_pred['Pred']<=threshold,0,1)
                        cm = pd.crosstab(Actual_pred['Actual'],Actual_pred['Pred'])
                        return(cm)
                    matrix_confusion=confusion_matrix_threshold(datay,average_scores,threshold)
                    
                    
                    #Metricas supervisadas
                    def Metricas_precision(Matriz):
                        # Cálculo de precisión
                        Precision = Matriz.iloc[0,0] / (Matriz.iloc[0,0] + Matriz.iloc[0,1])
                        
                        # Cálculo de exactitud
                        Exactitud = (Matriz.iloc[0,0] + Matriz.iloc[1,1]) / (Matriz.iloc[0,0] + Matriz.iloc[0,1] + Matriz.iloc[1,0] + Matriz.iloc[1,1])
                        
                        # Cálculo de especificidad
                        Especificidad = Matriz.iloc[1,1] / (Matriz.iloc[1,1] + Matriz.iloc[0,1])
                        
                        # Cálculo de la tasa de verdaderos positivos (TVP o Sensibilidad)
                        TVP = Matriz.iloc[0,0] / (Matriz.iloc[0,0] + Matriz.iloc[1,0])
                        
                        # Tasa de falsos negativos (TasaFN o FNR)
                        TasaFN = Matriz.iloc[1,0] / (Matriz.iloc[1,0] + Matriz.iloc[0,0])
                        
                        # Tasa de falsos positivos (TasaFP o FPR)
                        TasaFP = Matriz.iloc[0,1] / (Matriz.iloc[0,1] + Matriz.iloc[1,1])
                        
                        # Valor predictivo positivo (VPP o Precisión)
                        VPP = Matriz.iloc[0,0] / (Matriz.iloc[0,1] + Matriz.iloc[0,0])
                        
                        # Valor predictivo negativo (VPN)
                        VPN = Matriz.iloc[1,1] / (Matriz.iloc[1,1] + Matriz.iloc[1,0])
                        
                        # Cálculo de F1 Score
                        F1 = (2 * Precision * TVP) / (Precision + TVP)
                        return Precision, Exactitud, Especificidad, TVP, TasaFN, TasaFP, VPP, VPN, F1

                    Precision_n, Exactitud_n, Especificidad_n, TVP_n, TasaFN_n, TasaFP_n, VPP_n, VPN_n, F1_n=Metricas_precision(matrix_confusion)
                      
                    
                    Actual_predIF = pd.DataFrame({'Actual': datay, 'Pred':average_scores})
                    Actual_predIF['Pred'] = np.where(Actual_predIF['Pred']<=threshold,0,1)
                    actual=Actual_predIF['Actual']
                    pred=Actual_predIF['Pred']
                    
                    
                    with col2:
                        
                        
                    # Curva ROC
                        def plot_roc_curve(true_y, y_prob):
                            fpr, tpr, thresholds = roc_curve(true_y, y_prob)
                            roc_auc = auc(fpr, tpr)

                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=fpr,
                                y=tpr,
                                mode='lines',
                                name=f'ROC curve = {roc_auc:.2f}',
                                line=dict(color='red', width=2)
                            ))
                            fig.add_trace(go.Scatter(
                                x=[0, 1],
                                y=[0, 1],
                                mode='lines',
                                name='Random Guess',
                                line=dict(color='blue', dash='dash', width=2)
                            ))

                            fig.update_layout(
                                title="Curva ROC",
                                title_x=0.5,
                                xaxis=dict(
                                    title="Tasa de Falsos Positivos (FPR)",
                                    range=[0, 1],
                                    gridcolor='rgba(255,255,255,0.1)',
                                    linecolor='rgba(255,255,255,0.2)',
                                    tickcolor='rgba(255,255,255,0.5)',
                                    showgrid=True,
                                    zeroline=False
                                ),
                                yaxis=dict(
                                    title="Tasa de Verdaderos Positivos (TPR)",
                                    range=[0, 1],
                                    gridcolor='rgba(255,255,255,0.1)',
                                    linecolor='rgba(255,255,255,0.2)',
                                    tickcolor='rgba(255,255,255,0.5)',
                                    showgrid=True,
                                    zeroline=False
                                ),
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)',
                                font=dict(color='white', size=12),
                                legend=dict(
                                    yanchor="top",
                                    y=0.99,
                                    xanchor="left",
                                    x=0.01,
                                    bgcolor="rgba(0,0,0,0)"
                                )
                            )

                            st.plotly_chart(fig, use_container_width=True)

                        plot_roc_curve(actual, pred)
                        curve_roc = roc_auc_score(actual, pred)
                    
                    
                    # Explicación interactiva para la curva ROC
                    with st.expander("📖 ¿Qué es la Curva ROC?"):
                        
                    
                        st.markdown("""
                    La Curva ROC (Receiver Operating Characteristic) muestra la relación entre la tasa de verdaderos positivos 
                    (TPR) y la tasa de falsos positivos (FPR) en diferentes umbrales de clasificación. 
                    - AUC (Área bajo la curva): Indica la capacidad del modelo para distinguir entre las clases.
                    - Cuanto mayor sea el área bajo la curva (AUC), mejor será el rendimiento del model""")        
                        
                    # Mostrar métricas en cards
                    st.markdown("### 📊 Métricas Externas")
                    cols = st.columns(6)
                    with cols[0]:
                        st.metric(
                            "Precision",
                            f"{Precision_n:.2f}",
                            delta="Bueno" if Precision_n > 0.75 else "Regular"
                        )
                    with cols[1]:
                        st.metric(
                            "Exactitud",
                            f"{Exactitud_n:.1f}",
                            delta="Bueno" if Exactitud_n > 0.75 else "Regular"
                        )
                    with cols[2]:
                        st.metric(
                            "F1-SCORES",
                            f"{F1_n:.2f}",
                            delta="Bueno" if F1_n > 0.75 else "Regular"
                        )
                    with cols[3]:
                        st.metric(
                            "Curve ROC",
                            f"{curve_roc:.2f}",
                            delta="Bueno" if curve_roc > 0.75 else "Regular"
                        )
                    with cols[4]:
                        st.metric(
                            "Falsos positivos",
                            f"{TasaFP_n:.2f}",
                            delta="Bueno" if TasaFP_n < 0.3 else "Regular"
                        )
                    with cols[5]:
                        st.metric(
                            "Falsos Negativos",
                            f"{TasaFN_n:.2f}",
                            delta="Bueno" if TasaFN_n < 0.3 else "Regular"
                        )
                    
                
                        
            # Tabla de resultados detallados
            with st.expander("📑 Detalles de las Predicciones"):
                results_df = pd.DataFrame({
                    'ID': data.index,
                    'Anomaly Score': average_scores,
                    'Class': ['0' if x == 0 else '1' for x in y_pred]
                })
                st.dataframe(results_df, use_container_width=True)


# Control de navegación
if st.session_state.page == "home":
    show_home_page()
else:
    show_dashboard_page()

# Pie de página mejorado
st.markdown("---")
col1, col2, col3 = st.columns([1,2,1])
with col2:
    st.markdown(
        f"""
        <div style='text-align: center'>
            <p>🛡 {t('footer.title')}</p>
            <p>© 2024 Innovasic | UCC</p>
        </div>
        """,
        unsafe_allow_html=True,
    )