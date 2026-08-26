import hashlib
import secrets
import time
from contextlib import contextmanager
from typing import Iterator

import psycopg
from psycopg import Connection
from psycopg.errors import UniqueViolation

from audio_transcribator.config import settings


PASSWORD_ALGORITHM = "pbkdf2_sha256"
PASSWORD_ITERATIONS = 600_000


@contextmanager
def get_connection() -> Iterator[Connection]:
    with psycopg.connect(settings.database_url) as connection:
        yield connection


def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        PASSWORD_ITERATIONS,
    ).hex()
    return f"{PASSWORD_ALGORITHM}${PASSWORD_ITERATIONS}${salt}${digest}"


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        algorithm, iterations, salt, expected_digest = stored_hash.split("$", 3)
    except ValueError:
        return False

    if algorithm != PASSWORD_ALGORITHM:
        return False

    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        int(iterations),
    ).hex()
    return secrets.compare_digest(digest, expected_digest)


def wait_for_database() -> None:
    last_error: Exception | None = None
    for _ in range(settings.database_connect_retries):
        try:
            with get_connection():
                return
        except psycopg.OperationalError as exc:
            last_error = exc
            time.sleep(settings.database_connect_delay_seconds)

    raise RuntimeError("Database is unavailable") from last_error


def init_database() -> None:
    wait_for_database()
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    login VARCHAR(255) NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )

    create_user(settings.api_username, settings.api_password, ignore_existing=True)
    create_user(settings.creator_username, settings.creator_password, ignore_existing=True)


def create_user(login: str, password: str, ignore_existing: bool = False) -> bool:
    login = login.strip()
    if not login or not password:
        raise ValueError("Login and password are required")

    password_hash = hash_password(password)
    try:
        with get_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "INSERT INTO users (login, password_hash) VALUES (%s, %s)",
                    (login, password_hash),
                )
        return True
    except UniqueViolation:
        if ignore_existing:
            return False
        raise


def verify_user_credentials(login: str, password: str) -> bool:
    login = login.strip()
    if not login or not password:
        return False

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT password_hash FROM users WHERE login = %s", (login,))
            row = cursor.fetchone()

    if row is None:
        return False

    return verify_password(password, row[0])
