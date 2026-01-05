import streamlit as st


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
