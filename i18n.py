import json
import os
from functools import lru_cache

SUPPORTED_LANGS = {"es", "en"}
DEFAULT_LANG = "es"

_current_lang = DEFAULT_LANG

# Fallback mínimo embebido para no fallar si faltan archivos
_FALLBACK = {
    "es": {
        "app": {
            "title": "Sistema IDS IoT - Detección de Intrusiones",
            "welcome": "Bienvenido al Sistema de Detección de Intrusiones en Redes IoT",
            "language": "Idioma",
            "model_select": "Seleccione el modelo de detección",
            "metric_select": "Seleccione el tipo de métrica",
            "run": "Iniciar",
            "stop": "Detener",
            "capture": "Iniciar captura de paquetes",
            "upload_csv": "Subir archivo CSV",
            "results": "Resultados",
            "dashboard": "Panel de Control",
            "home": "Inicio"
        }
    },
    "en": {
        "app": {
            "title": "IDS for IoT - Intrusion Detection",
            "welcome": "Welcome to the IoT Network Intrusion Detection System",
            "language": "Language",
            "model_select": "Select detection model",
            "metric_select": "Select metric type",
            "run": "Run",
            "stop": "Stop",
            "capture": "Start packet capture",
            "upload_csv": "Upload CSV file",
            "results": "Results",
            "dashboard": "Dashboard",
            "home": "Home"
        }
    }
}


def _candidate_locale_dirs():
    # 1) Variable de entorno
    env_dir = os.getenv("I18N_LOCALES_DIR")
    if env_dir and os.path.isdir(env_dir):
        yield env_dir
    # 2) Junto al módulo
    yield os.path.join(os.path.dirname(__file__), "locales")
    # 3) Directorio de trabajo actual
    yield os.path.join(os.getcwd(), "locales")
    # 4) Un nivel arriba (por si se ejecuta desde subcarpetas)
    yield os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "locales"))


def set_language(lang: str):
    global _current_lang
    _current_lang = lang if lang in SUPPORTED_LANGS else DEFAULT_LANG


def get_language() -> str:
    return _current_lang


@lru_cache(maxsize=8)
def _load_locale(lang: str) -> dict:
    # Busca el archivo {lang}.json en los posibles directorios
    filename = f"{lang}.json"
    for d in _candidate_locale_dirs():
        path = os.path.join(d, filename)
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                break  # si falla lectura, usamos fallback
    # Si no existe, intenta con el idioma por defecto
    if lang != DEFAULT_LANG:
        return _load_locale(DEFAULT_LANG)
    # Fallback embebido si tampoco existe el del idioma por defecto
    return _FALLBACK.get(lang, _FALLBACK[DEFAULT_LANG])


def _deep_get(data: dict, dotted_key: str):
    cur = data
    for part in dotted_key.split('.'):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def t(key: str, **kwargs) -> str:
    lang = get_language()
    data = _load_locale(lang)
    value = _deep_get(data, key)
    if value is None and lang != DEFAULT_LANG:
        value = _deep_get(_load_locale(DEFAULT_LANG), key)
    if isinstance(value, str):
        try:
            return value.format(**kwargs)
        except Exception:
            return value
    return key
