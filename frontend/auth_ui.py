import streamlit as st
from pathlib import Path
import datetime as _dt
from backend import auth


def ensure_auth():
    """Inicializa autenticación y muestra UI si no está autenticado."""
    auth.init_db()

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.username = None

    if not st.session_state.authenticated:
        _login_ui()
        st.stop()


def _login_ui():
    logo_path = Path(__file__).resolve().parent.parent / "img" / "logo.png"
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
