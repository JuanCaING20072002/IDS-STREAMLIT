# Usa una imagen base de Python compatible con ARM (Raspberry Pi)
FROM python:3.11-slim-bullseye

# Instala dependencias del sistema necesarias para tshark y librerías científicas
RUN echo 'wireshark-common wireshark-common/install-setuid boolean false' | debconf-set-selections && \
    apt-get update && \
    apt-get install -y tshark build-essential libpcap-dev python3-dev python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Crea el directorio de la app
WORKDIR /app

# Copia los archivos del proyecto
COPY . /app

# Instala las dependencias de Python
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Da permisos para capturar paquetes (tshark necesita privilegios)
RUN groupadd -r wireshark && usermod -a -G wireshark root && chmod +x /usr/bin/dumpcap

# Expone el puerto de Streamlit
EXPOSE 8501

# Comando para iniciar la app
CMD ["streamlit", "run", "app2IPV2.py", "--server.port=8501", "--server.address=0.0.0.0"]