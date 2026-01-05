import io
from datetime import datetime
from fpdf import FPDF
import matplotlib.pyplot as plt
import tempfile
import os

# Función para crear el PDF del reporte de predicción
def generar_reporte_pdf(resumen, labels, values, descripcion_amenazas, fig=None, logo_path=None):
    """
    resumen: texto resumen de la predicción
    labels: lista de etiquetas de clases
    values: lista de conteos por clase
    descripcion_amenazas: dict {clase: descripcion}
    fig: matplotlib.figure (opcional, para incluir gráfica de barras)
    logo_path: ruta a imagen del logo (opcional)
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Encabezado con logo (si disponible)
    resolved_logo = None
    if logo_path and os.path.exists(logo_path):
        resolved_logo = logo_path
    else:
        # Intento de ruta relativa: ../img/logo.png (desde frontend/)
        base_dir = os.path.dirname(os.path.dirname(__file__))
        candidate = os.path.join(base_dir, 'img', 'logo.png')
        if os.path.exists(candidate):
            resolved_logo = candidate
    if resolved_logo:
        try:
            pdf.image(resolved_logo, x=12, y=10, w=30)
            pdf.ln(20)  # margen bajo el logo
        except Exception:
            pass

    # Título
    pdf.set_font("Arial", 'B', 18)
    pdf.cell(0, 15, "Reporte de Predicción IDS", ln=True, align='C')
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, f"Fecha: {datetime.now().strftime('%d/%m/%Y %H:%M')}", ln=True, align='R')
    pdf.ln(5)

    # Descripción breve
    pdf.set_font("Arial", '', 12)
    pdf.multi_cell(0, 10, resumen)
    pdf.ln(5)

    # Gráfica de barras
    if fig is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmpfile:
            fig.savefig(tmpfile.name, bbox_inches='tight')
            pdf.image(tmpfile.name, x=30, w=150)
            os.unlink(tmpfile.name)
        pdf.ln(5)

    # Tabla de resultados
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Resumen de Resultados:", ln=True)
    pdf.set_font("Arial", '', 12)
    for label, value in zip(labels, values):
        pdf.cell(0, 8, f"{label}: {value}", ln=True)
    pdf.ln(5)

    # Descripción de amenazas
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Descripción de Amenazas:", ln=True)
    pdf.set_font("Arial", '', 12)
    for label in labels:
        desc = descripcion_amenazas.get(label, "Sin descripción disponible.")
        pdf.multi_cell(0, 8, f"{label}: {desc}")
        pdf.ln(1)

    # Pie de página
    pdf.set_y(-30)
    pdf.set_font("Arial", 'I', 10)
    pdf.cell(0, 10, "Sistema de Detección de Anomalías IoT - Innovasic UCC", 0, 0, 'C')

    # Salida en memoria como bytes (compatibles con Streamlit)
    pdf_buf = pdf.output(dest='S')
    if isinstance(pdf_buf, (bytes, bytearray)):
        return io.BytesIO(pdf_buf)
    # Compatibilidad con posibles str en versiones antiguas
    return io.BytesIO(str(pdf_buf).encode('latin1'))

# Diccionario ejemplo de descripciones de amenazas
DESCRIPCIONES_AMENAZAS = {
    'Normal': (
        'Tráfico habitual sin indicios de riesgo. Recomendación: mantener el '
        'monitoreo regular para detectar cambios o picos inusuales.'
    ),
    'Anómalo': (
        'Actividad fuera de patrón que puede ser fallo de configuración o intento '
        'de ataque. Recomendación: revisar equipos/servicios recientes y registros.'
    ),
    'DDOS_TCP': (
        'Múltiples conexiones TCP intentan saturar un servicio. Medidas: activar '
        'SYN cookies, límites por IP y cerrar puertos innecesarios.'
    ),
    'DDOS_UDP': (
        'Alto volumen de tráfico UDP hacia puertos abiertos. Medidas: limitar UDP, '
        'permitir solo puertos necesarios y aplicar filtrado con el proveedor.'
    ),
    'Reconnaissance': (
        'Escaneos para descubrir servicios y debilidades. Medidas: ocultar '
        'servicios, reglas anti‑scan y revisar logs de seguridad.'
    ),
}
