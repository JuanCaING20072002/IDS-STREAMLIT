import sqlite3
from pathlib import Path
import os
import base64
import hashlib
import hmac
import secrets
from datetime import datetime, timedelta

DB_PATH = Path(__file__).parent / "auth.db"


def _get_conn():
    return sqlite3.connect(DB_PATH)


def init_db():
    conn = _get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS access_codes (
                code TEXT NOT NULL,
                expires_at TIMESTAMP NOT NULL
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def _pbkdf2_hash(password: str, iterations: int = 100_000) -> str:
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return f"pbkdf2_sha256${iterations}${base64.b64encode(salt).decode()}${base64.b64encode(dk).decode()}"


def _pbkdf2_verify(password: str, hash_str: str) -> bool:
    try:
        algo, iter_s, salt_b64, dk_b64 = hash_str.split("$")
        iterations = int(iter_s)
        salt = base64.b64decode(salt_b64)
        expected = base64.b64decode(dk_b64)
        dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
        return hmac.compare_digest(dk, expected)
    except Exception:
        return False


def get_admin_count() -> int:
    conn = _get_conn()
    try:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM users WHERE role = 'admin'")
        (count,) = cur.fetchone()
        return int(count)
    finally:
        conn.close()


def create_admin(username: str, password: str) -> tuple[bool, str]:
    if not username or not password:
        return False, "Usuario y contraseña son requeridos"
    pw_hash = _pbkdf2_hash(password)
    conn = _get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO users (username, password_hash, role) VALUES (?, ?, 'admin')",
            (username.strip(), pw_hash),
        )
        conn.commit()
        return True, "Administrador creado correctamente"
    except sqlite3.IntegrityError:
        return False, "El usuario ya existe"
    finally:
        conn.close()


def verify_login(username: str, password: str, code: str) -> tuple[bool, str]:
    if not username or not password or not code:
        return False, "Todos los campos son obligatorios"
    conn = _get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT password_hash, role FROM users WHERE username = ?",
            (username.strip(),),
        )
        row = cur.fetchone()
        if not row:
            return False, "Usuario o contraseña inválidos"
        pw_hash, role = row
        if not _pbkdf2_verify(password, pw_hash):
            return False, "Usuario o contraseña inválidos"

        # Validar código vigente
        now = datetime.utcnow()
        cur.execute("SELECT code, expires_at FROM access_codes ORDER BY expires_at DESC LIMIT 1")
        row = cur.fetchone()
        if not row:
            return False, "No hay código vigente. Genere uno e intente nuevamente."
        current_code, expires_at_str = row
        expires_at = datetime.strptime(expires_at_str, "%Y-%m-%d %H:%M:%S")
        if now > expires_at:
            return False, "El código ha expirado. Genere uno nuevo."
        if not hmac.compare_digest(current_code, code.strip()):
            return False, "Código inválido"
        return True, role
    finally:
        conn.close()


def generate_code(ttl_seconds: int = 300) -> tuple[str, datetime]:
    code = "".join(secrets.choice("0123456789") for _ in range(6))
    expires_at = datetime.utcnow() + timedelta(seconds=ttl_seconds)
    conn = _get_conn()
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM access_codes")
        cur.execute(
            "INSERT INTO access_codes (code, expires_at) VALUES (?, ?)",
            (code, expires_at.strftime("%Y-%m-%d %H:%M:%S")),
        )
        conn.commit()
    finally:
        conn.close()
    return code, expires_at


def get_current_code() -> tuple[str | None, datetime | None]:
    conn = _get_conn()
    try:
        cur = conn.cursor()
        cur.execute("SELECT code, expires_at FROM access_codes ORDER BY expires_at DESC LIMIT 1")
        row = cur.fetchone()
        if not row:
            return None, None
        code, expires_at_str = row
        return code, datetime.strptime(expires_at_str, "%Y-%m-%d %H:%M:%S")
    finally:
        conn.close()
