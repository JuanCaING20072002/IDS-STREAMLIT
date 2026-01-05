import streamlit as st


def setup_page_and_style():
    st.set_page_config(
        page_title="Sistema IDS IoT - Detección de Intrusiones",
        page_icon="🛡",
        layout="wide",
        initial_sidebar_state="expanded",
    )

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
