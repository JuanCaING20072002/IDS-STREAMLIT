import subprocess
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
from del_columns import del_columns
from sklearn.metrics import DistanceMetric
import sklearn.metrics._dist_metrics as _dm
from io import StringIO


# ...antes en el código carga la animación (si no lo has hecho)
def load_lottie_url(url: str):
    import requests
    r = requests.get(url)
    if r.status_code == 200:
        return r.json()
    return None

lottie_processing = load_lottie_url("https://assets1.lottiefiles.com/packages/lf20_usmfx6bp.json")


# Verificar permisos
def check_permissions():
    try:
        # Verificar si podemos ejecutar tcpdump
        subprocess.run(['/usr/bin/tcpdump', '--version'], 
                      check=True, 
                      stdout=subprocess.PIPE, 
                      stderr=subprocess.PIPE)
    except Exception as e:
        print("❌ Error de permisos:", str(e))
        print("Ejecuta estos comandos como solución:")
        print("sudo setcap cap_net_raw,cap_net_admin=eip /usr/bin/tcpdump")
        print("newgrp pcap")
        exit(1)

check_permissions()


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


def pred_threshold(
    score, threshold):
    score = pd.Series(score)  # ✅ Convertir a Series si es un array
    Actual_pred = pd.DataFrame({'Pred': score})  
    Actual_pred['Pred'] = np.where(Actual_pred['Pred']<=threshold,1,-1)
    Actual_pred=Actual_pred.reset_index(drop=True)
    return (Actual_pred)


# IMPORTANTE: st.set_page_config debe ser la primera llamada a Streamlit en el script
st.set_page_config(
    page_title="Sistema IDS IoT - Detección de Intrusiones",
    page_icon="🛡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Función para cargar animaciones Lottie
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        st.error(f"Error al cargar la animación desde {url}. Estado HTTP: {r.status_code}")
        return None
    return r.json()

# Animaciones
lottie_loading = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_szlepvdh.json")
lottie_how_it_works = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_yd8fbnml.json")
lottie_network = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_ggwq3ysg.json") 
lottie_success = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_ktwnwv5m.json")  
lottie_security = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_jcsfwbvi.json")  

# Verificar que las animaciones se han cargado correctamente
if lottie_loading is None or lottie_success is None:
    st.error("Error al cargar las animaciones Lottie. Por favor, verifica las URLs.")

# Inicializar el estado de la sesión para la navegación
if "page" not in st.session_state:
    st.session_state.page = "home"

#----------------------------------------------------  Estilos CSS --------------------------------------------------------------------------------------------------------------------------------
st.markdown("""
<style>
    body {
        background-color: #121212; /* Fondo oscuro */
        color: #ffffff; /* Texto blanco */
        font-family: 'Roboto', sans-serif;
    }
    .main-header {
        font-family: 'Roboto', sans-serif;
        background: linear-gradient(90deg, #1E3D59 0%, #2E5077 100%); /* Gradiente azul */
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 30px;
    }
    .metric-card {
        background: rgba(30, 50, 77, 0.8); /* Fondo azul oscuro */
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
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
        background-color: rgba(30, 50, 77, 0.8); /* Fondo azul oscuro */
        color: white;
        border: none;
        padding: 1rem;
        border-radius: 10px;
    }
    .stButton > button {
        background: rgba(30, 50, 77, 0.8); /* Fondo azul oscuro */
        border: none;
        color: #ffffff;
        padding: 10px 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3), 0 1px 3px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background: rgba(46, 80, 119, 0.8); /* Hover azul más claro */
        transform: translateY(-2px);
        box-shadow: 0 7px 14px rgba(0, 0, 0, 0.3), 0 3px 6px rgba(0, 0, 0, 0.2);
    }
    .stButton > button:active {
        transform: translateY(1px);
        box-shadow: 0 3px 8px rgba(0, 0, 0, 0.3), 0 1px 3px rgba(0, 0, 0, 0.2);
    }
    .dashboard-form {
        background: rgba(30, 50, 77, 0.8); /* Fondo azul oscuro */
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        max-width: 600px;
        margin: auto;
    }
    .notification {
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
        animation: fadeOut 2s forwards;
        animation-delay: 3.5s;
        background-color: #2E5077; /* Fondo azul */
        color: white;
    }
    @keyframes fadeOut {
        from {opacity: 1;}
        to {opacity: 0; height: 0; padding: 0; margin: 0;}
    }
    .streamlit-expanderHeader, .stTextInput > div > div > input {
        color: #ffffff !important;
        background-color: rgba(30, 50, 77, 0.8) !important; /* Fondo azul oscuro */
    }
    .stDataFrame {
        color: #ffffff;
        background-color: rgba(30, 50, 77, 0.8); /* Fondo azul oscuro */
    }
    .card {
        background: rgba(30, 50, 77, 0.8); /* Fondo azul oscuro */
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
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
    .stButton > button {
        background: #1E3D59; /* Fondo azul oscuro */
        border: none;
        color: #ffffff; /* Texto blanco */
        padding: 10px 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3), 0 1px 3px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background: #2E5077; /* Hover azul más claro */
        transform: translateY(-2px);
        box-shadow: 0 7px 14px rgba(0, 0, 0, 0.3), 0 3px 6px rgba(0, 0, 0, 0.2);
    }
    .stButton > button:active {
        transform: translateY(1px);
        box-shadow: 0 3px 8px rgba(0, 0, 0, 0.3), 0 1px 3px rgba(0, 0, 0, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------- INICIO PAGE----------------------------------------------------------------------------------------------------------------------------
# Navegación mejorada
with st.sidebar:
    st.image("https://img.icons8.com/color/48/000000/shield.png", width=50)
    selected = option_menu(
        "Navegación",
        ["Inicio", "Panel de Control"],
        icons=["house", "graph-up"],
        menu_icon="cast",
        default_index=0
    )
    st.session_state.page = "home" if selected == "Inicio" else "dashboard"

def show_home_page():
    st.title("🛡 Sistema de Detección de Intrusiones para IoT")
    if lottie_security:
        st_lottie(lottie_security, height=200)
    st.markdown('</div>', unsafe_allow_html=True)

    st.write("""
    Bienvenido al Sistema de Detección de Intrusiones en Redes IoT, 
    una solución avanzada diseñada para monitorear y detectar anomalías 
    en redes de IoT, brindando seguridad y confiabilidad sin precedentes. Nuestro sistema utiliza modelos 
    de vanguardia en detección de anomalías, como LOF (Local Outlier Factor), IForest (Isolation Forest) y KNN (K-Nearest Neighbors), 
    para analizar patrones complejos y alertar sobre posibles irregularidades en la red en tiempo real.
    """)

    col1, col2 = st.columns(2)
    
    with col1:  
        st.subheader("🎯 Objetivos del Sistema")
        st.write("""
        Nuestro sistema utiliza algoritmos avanzados de machine learning para:
        - 🔍 Detección en tiempo real de anomalías
        - 📊 Análisis predictivo de patrones
        - 🚫 Identificación de amenazas potenciales
        - 📈 Monitoreo continuo del rendimiento
        """)
        
    with col2:
        pass

    st.markdown("---")
    st.subheader("💬 Interpretación de Resultados")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        #### 🟢 Métricas de Validación
        - Silhouette Score: Mide la calidad de los clusters (ideal > 0.5)
        - Calinski Score: Evalúa la separación entre clusters
        - Davies Score: Indica la similitud dentro de clusters
        """)
    
    with col2:
        st.markdown("""
        #### 🔰 Clasificación de Anomalías
        - Normal: Tráfico de red esperado
        - Anómalo: Patrones sospechosos
        - Métrica Combinada: Evaluación holística
        """)
    
    with col3:
        st.markdown("""
        #### 📈 Indicadores de Rendimiento
        - Precisión: Exactitud de detección
        - Recall: Cobertura de detección
        - F1-Score: Balance precisión-recall
        """)

    st.markdown("---")
    st.subheader("🔎 Modelos de Detección de Anomalías")
    with st.expander("LOF (Local Outlier Factor)"):
        st.write("""
        El modelo LOF (Local Outlier Factor) es un algoritmo de detección de anomalías que identifica puntos de datos que se encuentran en regiones de baja densidad en comparación con sus vecinos. Es útil para detectar comportamientos inusuales en la red IoT.
        """)
    with st.expander("IForest (Isolation Forest)"):
        st.write("""
        El modelo IForest (Isolation Forest) es un algoritmo de detección de anomalías que utiliza árboles de aislamiento para identificar puntos de datos anómalos. Es eficiente y efectivo para detectar anomalías en grandes conjuntos de datos.
        """)
    with st.expander("KNN (K-Nearest Neighbors)"):
        st.write("""
        El modelo KNN (K-Nearest Neighbors) es un algoritmo de detección de anomalías que clasifica puntos de datos en función de la distancia a sus vecinos más cercanos. Es útil para detectar anomalías en datos de red IoT.
        """)

    st.markdown("---")
    st.subheader("📖 Método de Uso del Panel")
    st.write("""
    Sigue estos pasos para utilizar el panel de detección de anomalías:
    """)
    st.markdown("""
    <div class="carousel">
        <div class="carousel-item">
            <h4>Paso 1: Cargar Datos</h4>
            <p>📤 Carga tus datos de red IoT en formato CSV.</p>
        </div>
        <div class="carousel-item">
            <h4>Paso 2: Seleccionar Modelo</h4>
            <p>📌 Selecciona el modelo de detección (LOF, IForest o KNN).</p>
        </div>
        <div class="carousel-item">
            <h4>Paso 3: Ejecutar Análisis</h4>
            <p>🖱 Ejecuta el análisis con un solo clic.</p>
        </div>
        <div class="carousel-item">
            <h4>Paso 4: Visualizar Resultados</h4>
            <p>📊 Visualiza los resultados en gráficos interactivos.</p>
        </div>
        <div class="carousel-item">
            <h4>Paso 5: Obtener Insights</h4>
            <p>📄 Obtén insights detallados sobre las anomalías detectadas.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("📊 Tipos de Resultados y Recomendaciones")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="card">
            <h4>✅ Resultados Normales</h4>
            <p>Los datos han sido clasificados como normales. No se han detectado anomalías significativas.</p>
            <p><strong>Recomendación:</strong> Continúa monitoreando la red regularmente.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="card">
            <h4>⚠ Resultados Anómalos</h4>
            <p>Se han detectado datos anómalos en la red. Esto puede indicar posibles amenazas o comportamientos inusuales.</p>
            <p><strong>Recomendación:</strong> Investiga las anomalías detectadas y toma las medidas necesarias para mitigar posibles riesgos.</p>
        </div>
        """, unsafe_allow_html=True)

#---------------------------------------------------- DASHBOARD ----------------------------------------------------------------------------------------------------------------------------



# DistanceMetric.get_metric("euclidean") devuelve un objeto cuya clase interna
# es la que Python debe encontrar al desempaquetar el pickle.
# Así que extraemos la __class__ y la asignamos al nombre que pickle busca:
_dm.EuclideanDistance = DistanceMetric.get_metric("euclidean").__class__

def show_dashboard_page():
    st.title('❇ Panel de Control de Detección')


    model_path = "model/lof__EJERCICIO10_n_components6_fit_over.pkl"
    loaded_object = joblib.load(model_path)
    print(f"Tipo del objeto cargado: {type(loaded_object)}")
    print(f"Contenido: {loaded_object}")
    
    model_path = "model/iforest__EJERCICIO10_n_components6_fit.pkl"
    loaded_object = joblib.load(model_path)
    print(f"Tipo del objeto cargado: {type(loaded_object)}")
    print(f"Contenido: {loaded_object}")
    # Rutas a los modelos
         
    LOF_MODEL_PATHS = "model/lof__EJERCICIO10_n_components6_fit_over.pkl"
    IFOREST_MODEL_PATHS = "model/iforest__EJERCICIO10_n_components6_fit.pkl"
   

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
    
     # Interfas Uusuario Panel de Control
    with col1:
        st.markdown("### 🖥 Panel de Control")
        model_option = st.selectbox(
            "📌 Seleccione el modelo de detección",
            ('LOF','IFOREST'),
            help="Cada modelo utiliza diferentes técnicas para detectar anomalías"
        )
        metric_option = st.selectbox(
            "📌 Seleccione el tipo de metrica",
            ('Externas', 'Internas'),
            help="Cada modelo utiliza diferentes técnicas para detectar anomalías"
        )

        
        
        # Función para procesar los paquetes capturados
        # ----------------------------------------------
        # Configuración del comando para Tshark
        comm_arg = (
    "sudo /usr/local/bin/tshark "
    "-i eth0 -i wlan0 "
    "-l -c 1000 "
    # filtro de captura:
    "-f \"tcp or udp or arp\" "
     "-f \"arp or (udp and not port 53 and not port 5353) or (tcp and not port 443)\" "
    "-T fields -E separator=/t "
    "-e frame.time_delta "
    "-e _ws.col.Protocol "
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
    "-e tcp.flags.syn -e tcp.flags.fin -e ip.version"
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

            # Mapeo de ARP al campo IP src y dst
            if packet_[1] == 'TCP':
                tcp_src = packet_[8]  # campo tcp.srcport
                tcp_dst = packet_[9]  # campo tcp.dstport
                        # Extraer puertos según el protocolo
            if packet_[1] == 'TCP':
                tcp_src = packet_[8]  # campo tcp.srcport
                tcp_dst = packet_[9]  # campo tcp.dstport
            elif packet_[1] == 'UDP':
                tcp_src = packet_[10]  # reasignación de udp.srcport a tcp.srcport
                tcp_dst = packet_[11]  # reasignación de udp.dstport a tcp.dstport
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
            ordered_packet = [
                packet_[0], packet_[1],tcp_src, tcp_dst, packet_[10], packet_[11], packet_[14],
                packet_[15], packet_[16], packet_[17], packet_[18], packet_[19], packet_[20], packet_[21],
                packet_[22], packet_[23], packet_[24], packet_[25], packet_[26], packet_[27], packet_[28],
                packet_[29], packet_[30], packet_[31], packet_[33]
            ]

            fieldnames = [
                'delta_time', 'protocols', 'port_src', 'port_dst', 'frame_len',
                'udp_len', 'ip_ttl', 'icmp_type', 'tos', 'ip_flags_rb', 'ip_flags_df', 'ip_flags_mf',
                'tcp_flags_res', 'tcp_flags_ns', 'tcp_flags_cwr', 'tcp_flags_ecn', 'tcp_flags_urg',
                'tcp_flags_ack', 'tcp_flags_push', 'tcp_flags_reset', 'tcp_flags_syn', 'tcp_flags_fin',
            ]

            return {fieldnames[i]: [ordered_packet[i]] for i in range(len(fieldnames))}


        # Función para agregar los paquetes al DataFrame
        def packet_df(type_packet, df):
            if df is None:
                df = pd.DataFrame(type_packet)
            else:
                df = pd.concat([df, pd.DataFrame(type_packet)], axis=0, ignore_index=True)
                
                
            return df
        
        def predecir(model, data):
            try:
                
                # Obtener las columnas desde el preprocesador
                columns = model[0].named_steps['prepro_2_del']\
                  .named_steps['prepro_1_num_cat']\
                  .get_feature_names_out().tolist()

                # Reemplazar del_columns en el pipeline con la versión correcta
                model[0].named_steps['prepro_2_del'].named_steps['del_columns'] = del_columns(columns)
                #'category__protocols'
                columnas=['number__delta_time',	'number__tcp_srcport',	'number__tcp_dstport',	'number__frame_len',	'number__udp_length',	'number__ip_ttl',	'number__tos',	'number__ip_flags_df',	'number__ip_flags_mf',	'number__tcp_flags_ns',	'number__tcp_flags_syn']

                columnas_pca=['componente1','componente2']

                columnas_pca2=['componente1','componente2','componente3','componente4','componente5','componente6']


                scores = model.decision_function(data)
                predicciones=pred_threshold(scores, -0.15)
                                
                pp3 = model[0].transform(data)  # Transformar los datos para el modelo IForest
                
                pp3 = pd.DataFrame(pp3, columns=columnas_pca2)  # Convertir a DataFrame
                
                # Añadir la columna de cluster al DataFrame
                pp3['cluster'] = predicciones
                
                
                return pp3, predicciones
                
            except Exception as e:
                st.error(f"Error en la predicción: {str(e)}")
                import traceback
                st.error(f"Detalles: {traceback.format_exc()}")
                return None, None
        
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
                    f"{calinski:.1f}",
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
            
        # Streamlit UI
        st.title("🔍 Captura y Análisis de Tráfico de Red")

        traffic_method = st.selectbox(
            "📌 Seleccione el método de tráfico",
            ('Iniciar captura de paquetes tiempo real', 'Abrir captura de paquetes pcap')
        )

        data = None  # Inicializamos el DataFrame

        if traffic_method == 'Iniciar captura de paquetes tiempo real':
            import requests

            def fetch_packets_from_api(api_url):
                try:
                    response = requests.get(api_url)
                    response.raise_for_status()
                    data = response.json()
                    return data.get("packets", [])
                except Exception as e:
                    st.error(f"Error al conectar con la API: {str(e)}")
                    return []

            api_url = "http://192.168.1.39:5000/capture/tshark"  # IP de tu Raspberry Pi
            if st.button("🚀 Iniciar Captura"):
                st.write("📡 Capturando paquetes desde la Raspberry Pi...")
                packets = fetch_packets_from_api(api_url)
                if packets:
                    captured_data = pd.DataFrame(packets, columns=["delta_time", "protocols", "ip_src", "ip_dst", "tcp_srcport", "tcp_dstport", "frame_len"])
                    st.write("📊 **Paquetes capturados:**")
                    st.dataframe(captured_data)
                else:
                    st.error("No se capturaron paquetes.")

        elif traffic_method == 'Abrir captura de paquetes pcap':
            st.markdown("### 📂 Capturar y Descargar archivo PCAP")
    
            if st.button("🚀 Iniciar Captura de PCAP"):
                try:
                    # Crear archivo temporal con nombre único
                    pcap_path = f"captura_{int(time.time())}.pcap"

                    # Filtro de captura para tcpdump
                    bpf_filter = 'tcp or udp or arp'


                    capture_cmd = [
                        '/usr/bin/tcpdump',
                        '-i', 'wlan0',
                        '-w', pcap_path,
                        '-c', '100',
                        '-q',
                        bpf_filter
                    ]

                    # Mostrar spinner + barra de progreso simulada
                    with st.spinner("📡 Capturando paquetes..."):
                        progress_bar = st.progress(0)
                        total_packets = 100

                        # Simulación de avance mientras corre tcpdump
                        for i in range(total_packets):
                            time.sleep(0.05)  # Simula avance mientras tcpdump trabaja (~5 seg)
                            progress_bar.progress((i + 1) / total_packets)

                        process = subprocess.run(
                            capture_cmd,
                            timeout=60,
                            check=True,
                            capture_output=True,
                            text=True
                        )

                    if not os.path.exists(pcap_path) or os.path.getsize(pcap_path) == 0:
                        st.error("❌ El archivo PCAP está vacío")
                        if process.stderr:
                            st.error(f"Error: {process.stderr}")
                        return

                    st.success(f"✅ Captura completada. Tamaño: {os.path.getsize(pcap_path)} bytes")
            
                    # Proporcionar el archivo para descarga
                    with open(pcap_path, "rb") as file:
                        st.download_button(
                            label="📥 Descargar archivo PCAP",
                            data=file,
                            file_name=os.path.basename(pcap_path),
                            mime="application/octet-stream"
                        )

                except subprocess.TimeoutExpired:
                    st.error("🕒 Tiempo de captura excedido")
                except Exception as e:
                    st.error(f"❌ Error inesperado: {str(e)}")
                    
            else:
                # Mantener la parte de carga de archivo PCAP original
                st.markdown("### 📂 Cargar archivo PCAP")
                
                uploaded_pcap = st.file_uploader(
                    "📂 Cargar archivo PCAP",
                    type=["pcap", "pcapng"],
                    help="Cargue un archivo de captura de paquetes (PCAP/PCAPNG)",
                    key="pcap_uploader"
                )
                
                if uploaded_pcap is not None:
                    # Guardamos temporalmente el archivo PCAP
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pcap') as tmp_file:
                        tmp_file.write(uploaded_pcap.getvalue())
                        pcap_path = tmp_file.name
                    
                    st.success(f"✅ Archivo PCAP cargado correctamente")

                   
                    
                    if st.button("🔍 Procesar archivo PCAP"):
                        with st.spinner("📡 Procesando archivo PCAP..."):

                            if lottie_processing:
                                st_lottie(lottie_processing, height=200, key="processing_animation")
                            
                            # Definimos los campos a extraer con tshark
                            tshark_fields = [
                                'frame.time_delta', '_ws.col.Protocol', 'ip.src', 'ip.dst',
                                'arp.src.proto_ipv4', 'arp.dst.proto_ipv4', 'ipv6.src', 'ipv6.dst',
                                'eth.src', 'eth.dst', 'tcp.srcport', 'tcp.dstport', 'udp.srcport',
                                'udp.dstport', 'frame.len', 'udp.length', 'ip.ttl', 'icmp.type',
                                'ip.dsfield.dscp', 'ip.flags.rb', 'ip.flags.df', 'ip.flags.mf',
                                'tcp.flags.res', 'tcp.flags.ns', 'tcp.flags.cwr', 'tcp.flags.ecn',
                                'tcp.flags.urg', 'tcp.flags.ack', 'tcp.flags.push', 'tcp.flags.reset',
                                'tcp.flags.syn', 'tcp.flags.fin', 'ip.version'
                            ]
                            
                            # Construimos el comando tshark
                            tshark_cmd = [
                                'tshark',
                                '-r', pcap_path,  # Archivo de entrada
                                '-T', 'fields',
                                '-E', 'separator=\t'
                            ]
                            
                            # Agregamos cada campo con la opción -e
                            for field in tshark_fields:
                                tshark_cmd.extend(['-e', field])
                            
                            # Ejecutamos tshark y capturamos la salida
                            try:
                                process = subprocess.Popen(tshark_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                                
                                # Lista para almacenar todos los paquetes procesados
                                all_packets = []
                                
                                # Procesamos cada línea (cada paquete)
                                for line in process.stdout:
                                    # Separamos los campos por tabulación
                                    packet_ = line.rstrip('\n').split('\t')
                                    
                                    # Aseguramos que el paquete tenga 34 campos
                                    if len(packet_) < 34:
                                        packet_ += [''] * (34 - len(packet_))
                                    
                                    # Aplicamos el mismo procesamiento que en la función type_packet
                                    # para mantener la coherencia con el resto del sistema
                                    if packet_[1] == 'TCP':
                                        tcp_src = packet_[8]  # campo tcp.srcport
                                        tcp_dst = packet_[9]  # campo tcp.dstport
                                    elif packet_[1] == 'UDP':
                                        tcp_src = packet_[10]  # reasignación de udp.srcport a tcp.srcport
                                        tcp_dst = packet_[11]  # reasignación de udp.dstport a tcp.dstport
                                    else:
                                        tcp_src = ''
                                        tcp_dst = ''
                                            
                                    if packet_[1] == 'ARP':
                                        packet_[2] = packet_[4]  # ip.src <- arp.src.proto_ipv4
                                        packet_[3] = packet_[5]  # ip.dst <- arp.dst.proto_ipv4
                                    
                                    # Mapeo de IPv6
                                    if packet_[32] == '6':
                                        packet_[2] = packet_[6]  # ip.src <- ipv6.src
                                        packet_[3] = packet_[7]  # ip.dst <- ipv6.dst
                                    
                                    # Reemplazo de comas por puntos
                                    for index in [16, 18, 19, 20, 21]:
                                        if packet_[index]:
                                            packet_[index] = packet_[index].replace(',', '.')
                                    
                                    # Estructura del paquete ordenado siguiendo el formato de type_packet
                                    ordered_packet = {
                                        'delta_time': [packet_[0]], 
                                        'protocols': [packet_[1]],
                                        'tcp_srcport': [tcp_src], 
                                        'tcp_dstport': [tcp_dst], 
                                        'frame_len': [packet_[14]],
                                        'udp_length': [packet_[15]], 
                                        'ip_ttl': [packet_[16]], 
                                        'icmp_type': [packet_[17]], 
                                        'tos': [packet_[18]], 
                                        'ip_flags_rb': [packet_[19]], 
                                        'ip_flags_df': [packet_[20]], 
                                        'ip_flags_mf': [packet_[21]],
                                        'tcp_flags_res': [packet_[22]], 
                                        'tcp_flags_ns': [packet_[23]], 
                                        'tcp_flags_cwr': [packet_[24]], 
                                        'tcp_flags_ecn': [packet_[25]], 
                                        'tcp_flags_urg': [packet_[26]],
                                        'tcp_flags_ack': [packet_[27]], 
                                        'tcp_flags_push': [packet_[28]], 
                                        'tcp_flags_reset': [packet_[29]], 
                                        'tcp_flags_syn': [packet_[30]], 
                                        'tcp_flags_fin': [packet_[31]]
                                    }
                                    
                                    # Agregar al DataFrame con la función packet_df
                                    # pero primero creamos el DataFrame del paquete individual
                                    packet_df_single = pd.DataFrame(ordered_packet)
                                    
                                    # Agregamos al DataFrame acumulativo
                                    if not all_packets:
                                        all_packets.append(packet_df_single)
                                    else:
                                        all_packets.append(packet_df_single)
                                
                                # Concatenamos todos los DataFrames
                                if all_packets:
                                    captured_data = pd.concat(all_packets, ignore_index=True)
                                    
                                    # Convertimos columnas numéricas
                                    numeric_columns = ['delta_time', 'frame_len', 'udp_length', 'ip_ttl']
                                    for col in numeric_columns:
                                        if col in captured_data.columns:
                                            captured_data[col] = pd.to_numeric(captured_data[col], errors='coerce')
                                    
                                    # Limpiamos el archivo temporal
                                    os.unlink(pcap_path)
                                    
                                    # Mostramos el DataFrame
                                    st.write(f"📊 *Se procesaron {len(captured_data)} paquetes:*")
                                    st.dataframe(captured_data)
                                    
                                    # Guardamos en la sesión
                                    st.session_state.data = captured_data
                                    
                                    st.success("✅ Datos del PCAP listos para predicción")
                                    
                                    # Botón para guardar en CSV
                                    if st.button("💾 Guardar en CSV"):
                                        save_packet_csv(captured_data, "pcap_procesado.csv")
                                else:
                                    st.error("❌ No se pudo extraer ningún paquete del archivo PCAP")
                                    
                            except Exception as e:
                                st.error(f"❌ Error al procesar el archivo PCAP: {str(e)}")
                                import traceback
                                st.error(f"Detalles: {traceback.format_exc()}")
                                try:
                                    os.unlink(pcap_path)  # Intentar limpiar el archivo temporal
                                except:
                                    pass
        # Sección de carga de archivo CSV
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

            n_normales = 0
            n_anomalos = 0
            pp3 = None
            y_pred = None

                
            if data is not None:
                loading_placeholder = st.empty()
                with loading_placeholder.container():
                    if lottie_loading is not None:
                        st_lottie(lottie_loading, height=200, key="loading")
                
                try:
                    # Inicializar el modelo
                    model = None
                    
                     # Cargar el modelo según la opción seleccionada
                    if model_option == 'LOF':
                        model = load_model(LOF_MODEL_PATHS)
                    elif model_option == 'IFOREST':
                        model = load_model(IFOREST_MODEL_PATHS)
                        if model is None:
                            raise Exception("No se pudo cargar el modelo Seleccionado")
                
                    
                    if traffic_method == 'Iniciar captura de paquetes tiempo real':
                                  
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
                        pp3, y_pred = predecir(model, data)
                        
                        n_normales  = int((y_pred == 1).sum())
                        n_anomalos  = int((y_pred == -1).sum())
                        
                        st.write(n_normales,n_anomalos)
                        
                        
                        
                        if pp3 is None or y_pred is None:
                            raise Exception("Error en la predicción")
                    
                    
                    if y_pred is not None and n_normales > 0 and n_anomalos > 0:
 
                        # Calcular métricas internas
                        silhouette, calinski, davies = metricas.metrica_internas(pp3, y_pred)
                        
                        if metric_option == 'Internas':   
                            mostrar_metricas(silhouette, calinski, davies)        
                    else:
                        st.write("No se pueden calcular las métricas porque solo hay un clúster")
                                        
                    
                    
                    # Limpiar animación de carga
                    time.sleep(3)
                    loading_placeholder.empty()
                    
                    # Sección de visualización
                    with col1:
                        # Contar normales y anómalos usando y_pred
                        
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
                                title="Clasificación",
                                showgrid=False,
                                showline=True,
                                linecolor='rgba(255,255,255,0.2)'
                            ),
                            yaxis=dict(
                                title="Cantidad",
                                showgrid=True,
                                gridcolor='rgba(255,255,255,0.1)',
                                zeroline=False
                            ),
                            bargap=0.2
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
            # Aquí continuarías con las visualizaciones...
                    # Resto del código para visualizaciones
                    # ...
                
                except Exception as e:
                    st.error(f"❌ Error en el procesamiento: {str(e)}")
                    import traceback
                    st.error(f"Detalles: {traceback.format_exc()}")
                    loading_placeholder.empty()
            else:
                st.warning("⚠️ Por favor capture paquetes o cargue un archivo CSV antes de realizar la predicción.")
        
        uploaded_file = st.file_uploader(
            "📂 Cargar archivo CSV",
            type="csv",
            help="Cargue sus datos de red en formato CSV"
        )
    

    with col2:
        if uploaded_file is not None:
            with st.spinner('⏳ Procesando datos...'):
                data = pd.read_csv(uploaded_file)
                st.markdown('<div class="notification" style="background-color: #4CAF50; color: white;">✅ Datos cargados exitosamente</div>', unsafe_allow_html=True)
                time.sleep(3)

                with st.expander("📊 Vista previa de datos"):
                    st.dataframe(data.head())
                    st.info(f"Dimensiones del dataset: {data.shape[0]} filas, {data.shape[1]} columnas")

    if uploaded_file is not None and st.button('🚀 Realizar Predicción', key='predict'):
        loading_placeholder = st.empty()
        cancel_button = st.empty()
        with loading_placeholder.container():
            if lottie_loading is not None:
                st_lottie(lottie_loading, height=200, key="loading")
                cancel_button.button("Cancelar Predicción", key='cancel')
        

        
            if model_option == 'IFOREST':
                # Cargar directamente el modelo OCSVM
                model = load_model(IFOREST_MODEL_PATHS)
                current_models = None  # No necesitamos current_models aquí
            else:
                # Seleccionar rutas para los otros modelos
                if model_option == 'LOF':
                    MODEL_PATHS = LOF_MODEL_PATHS
                elif model_option == 'IFOREST':
                    MODEL_PATHS = IFOREST_MODEL_PATHS

            # Cargar modelos desde las rutas
                current_models = load_models(MODEL_PATHS)
               
            # Procesamiento de datos para otros modelos
             
        #    for i, (key, model) in enumerate(current_models.items()):  
                
               
                if traffic_method == 'Iniciar captura de paquetes tiempo real' or traffic_method == 'Abrir captura de paquetes pcap':
                
                    pp3,y_pred=predecir(model,captured_data) 
                    
                    silhouette,calinski,davies = metricas.metrica_internas(pp3,pp3['cluster'])  
                    
                    mostrar_metricas(silhouette,calinski,davies)   
                
                    
            # Limpiar animación de carga después de 3 segundos
            time.sleep(3)
            loading_placeholder.empty()
    
    
            
                
            # Mostrar animación de éxito
            with st.container():
                if lottie_success:
                    st_lottie(lottie_success, height=200, key="success", speed=1.5)
                    time.sleep(3)

            # Resultados y visualizaciones
             # Visualization section
            # Sección de visualización
            col1, col2 = st.columns(2)
            

        with col1:
            
            num_clusters = len(np.unique(pp3['cluster']))
                            # Cálculo de las cantidades de normales y anómalos

            if num_clusters > 1:
                normal_count = np.sum(pp3['cluster'] ==0 )
                anomaly_count = np.sum(pp3['cluster'] == 1)
            else:
                normal_count = np.sum(pp3['cluster'] == 0)
                anomaly_count = 0
            st.write(normal_count, anomaly_count)    

            # Crear el gráfico de barras con Plotly
            fig = go.Figure(data=[
                go.Bar(
                    x=['Normal', 'Anómalo'],  # Etiquetas del eje X
                    y=[normal_count, anomaly_count],  # Valores del eje Y
                    marker_color=['#00C853', '#FF5252'],  # Colores personalizados
                    text=[f"{normal_count:,}", f"{anomaly_count:,}"],  # Texto en las barras
                    textposition='auto',  # Posición del texto
                    hoverinfo='y+text',  # Información al pasar el mouse
                    width=0.8  # Ancho de las barras
                )
            ])

            # Configuración de diseño del gráfico
            fig.update_layout(
                title="Distribución de Detecciones",  # Título del gráfico
                title_x=0.5,  # Centrar el título
                plot_bgcolor='rgba(0,0,0,0)',  # Fondo transparente del área de trazado
                paper_bgcolor='rgba(0,0,0,0)',  # Fondo transparente del gráfico
                font=dict(color='white'),  # Color del texto
                showlegend=False,  # Ocultar leyenda
                margin=dict(t=50, l=50, r=50, b=50),  # Márgenes del gráfico
                width=800,  # Ancho total del gráfico
                height=700,  # Altura total del gráfico
                xaxis=dict(
                    title="Clasificación",  # Título del eje X
                    showgrid=False,  # Ocultar la cuadrícula del eje X
                    showline=True,  # Mostrar línea del eje X
                    linecolor='rgba(255,255,255,0.2)'  # Color de la línea del eje X
                ),
                yaxis=dict(
                    title="Cantidad",  # Título del eje Y
                    showgrid=True,  # Mostrar cuadrícula del eje Y
                    gridcolor='rgba(255,255,255,0.1)',  # Color de la cuadrícula del eje Y
                    zeroline=False  # Ocultar línea en Y=0
                ),
                bargap=0.2  # Espacio entre las barras
            )

            # Mostrar el gráfico en Streamlit
            st.plotly_chart(fig, use_container_width=True)  # Permitir que se ajuste al contenedor
                
                
            
            # Matriz de Confusión
            
            def confusion_matrix_threshold(actual,score, threshold):
                Actual_pred = pd.DataFrame({'Actual': actual, 'Pred': score})
                Actual_pred['Pred'] = np.where(Actual_pred['Pred']<=threshold,0,1)
                cm = pd.crosstab(Actual_pred['Actual'],Actual_pred['Pred'])
                return(cm)
            
            if(metric_option=='Externas'):
                with st.expander("📖 Explicación de las métricas Externas"):
                        
                        st.write("""
                        1. Precisión (Precision):
                        - Indica qué porcentaje de las predicciones positivas realizadas por el modelo son correctas.
                        - Fórmula: TP / (TP + FP)
                        - Ejemplo: Si el modelo predice 100 positivos y 98 son correctos, la precisión es 0.98.

                        2. Exactitud (Accuracy):
                        - Porcentaje de predicciones correctas sobre el total de predicciones realizadas.
                        - Fórmula: (TP + TN) / (TP + TN + FP + FN)
                        - Muestra qué tan bien el modelo clasifica en general.

                        3. F1-Score:
                        - Es la media armónica entre la Precisión y la Sensibilidad (Recall).
                        - Útil cuando las clases están desbalanceadas.
                        - Fórmula: 2 * (Precision * Recall) / (Precision + Recall)

                        4. Curva ROC y AUC:
                        - La curva ROC muestra la relación entre la Tasa de Verdaderos Positivos (TPR) y la Tasa de Falsos Positivos (FPR).
                        - El AUC (Área Bajo la Curva) mide qué tan bien el modelo separa las clases.
                        - Un valor cercano a 1 indica un excelente desempeño; 0.76 es aceptable.

                        5. Falsos Positivos (FP):
                        - Representan los casos negativos que el modelo clasificó erróneamente como positivos.
                        - Fórmula: FP / (FP + TN)
                        - Menos falsos positivos indican mejor rendimiento.

                        6. Falsos Negativos (FN):
                        - Representan los casos positivos reales que el modelo no detectó correctamente.
                        - Fórmula: FN / (FN + TP)
                        - Un valor bajo indica que el modelo es bueno para detectar positivos reales.

                        Nota:
                        - TP: Verdaderos Positivos
                        - TN: Verdaderos Negativos
                        - FP: Falsos Positivos
                        - FN: Falsos Negativos
                        """)    
                st.markdown("### 📊 Matriz de Confusión")
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
                
            if(metric_option=='Internas'):
                with st.expander("📖 Explicación de las métricas Internas"):
                        
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
                
            with col2:
                
                # Métricas de rendimiento
                X_test_norm = data.values
                if metric_option== 'Externas':
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
                    'Score de Anomalía': average_scores,
                    'Clasificación': ['0' if x == 0 else '1' for x in y_pred]
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
        """
        <div style='text-align: center'>
            <p>🛡 Sistema de Detección de Anomalías IoT</p>
            <p>© 2025 Innovasic | UCC</p>
        </div>
        """,
        unsafe_allow_html=True)