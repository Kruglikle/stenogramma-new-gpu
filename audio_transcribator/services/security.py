import ipaddress
import re
import socket
from pathlib import Path
from urllib.parse import urlparse

from audio_transcribator.config import settings


LOCAL_HOSTNAMES = {"localhost", "localhost.localdomain"}
SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize_upload_filename(filename: str | None) -> str:
    raw_name = Path(filename or "upload").name
    safe_name = SAFE_FILENAME_RE.sub("_", raw_name).strip("._")
    return safe_name or "upload"


def _is_blocked_ip(address: str) -> bool:
    ip = ipaddress.ip_address(address)
    return any(
        (
            ip.is_private,
            ip.is_loopback,
            ip.is_link_local,
            ip.is_multicast,
            ip.is_reserved,
            ip.is_unspecified,
        )
    )


def _resolve_host_addresses(hostname: str, port: int) -> set[str]:
    try:
        infos = socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise ValueError("Could not resolve media URL host") from exc
    return {info[4][0] for info in infos}


def validate_public_media_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or not parsed.hostname:
        raise ValueError("Only http/https media links are supported")
    if parsed.username or parsed.password:
        raise ValueError("Media URLs with embedded credentials are not allowed")

    if settings.allow_private_media_urls:
        return url

    hostname = parsed.hostname.strip().lower()
    if hostname in LOCAL_HOSTNAMES:
        raise ValueError("Private or local media URLs are not allowed")

    try:
        if _is_blocked_ip(hostname):
            raise ValueError("Private or local media URLs are not allowed")
    except ValueError:
        addresses = _resolve_host_addresses(hostname, parsed.port or (443 if parsed.scheme == "https" else 80))
        if any(_is_blocked_ip(address) for address in addresses):
            raise ValueError("Private or local media URLs are not allowed")

    return url
