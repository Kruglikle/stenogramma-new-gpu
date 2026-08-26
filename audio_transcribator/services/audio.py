import mimetypes
import re
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen

from audio_transcribator.services.security import validate_public_media_url


VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus", ".aac"}
MEDIA_EXTENSIONS = VIDEO_EXTENSIONS | AUDIO_EXTENSIONS
CONTENT_TYPE_EXTENSIONS = {
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/mp4": ".m4a",
    "audio/x-m4a": ".m4a",
    "audio/flac": ".flac",
    "audio/ogg": ".ogg",
    "video/mp4": ".mp4",
    "video/quicktime": ".mov",
    "video/x-msvideo": ".avi",
    "video/x-matroska": ".mkv",
    "video/webm": ".webm",
}


def _safe_url_filename(url: str, content_type: str | None) -> str:
    parsed = urlparse(url)
    raw_name = Path(unquote(parsed.path)).name
    sanitized_name = re.sub(r"[^A-Za-z0-9._-]+", "_", raw_name).strip("._")
    suffix = Path(sanitized_name).suffix.lower()

    if suffix not in MEDIA_EXTENSIONS:
        suffix = CONTENT_TYPE_EXTENSIONS.get((content_type or "").split(";")[0].strip().lower(), "")
        sanitized_name = f"remote_media{suffix}"

    if not suffix:
        raise ValueError("Could not determine media format from URL or Content-Type")

    return sanitized_name or f"remote_media{suffix}"


def _content_disposition_filename(content_disposition: str | None) -> str | None:
    if not content_disposition:
        return None

    match = re.search(r'filename\*?=(?:UTF-8\'\')?"?([^";]+)"?', content_disposition, flags=re.IGNORECASE)
    if not match:
        return None

    return unquote(match.group(1)).strip()


def download_media(url: str, output_dir: Path) -> Path:
    url = validate_public_media_url(url)

    try:
        return _download_direct_media(url, output_dir)
    except Exception as direct_error:
        return _download_with_yt_dlp(url, output_dir, direct_error)


def _download_direct_media(url: str, output_dir: Path) -> Path:
    request = Request(url, headers={"User-Agent": "audio-transcribator/1.0"})
    try:
        with urlopen(request, timeout=60) as response:
            validate_public_media_url(response.geturl())
            content_type = response.headers.get("Content-Type")
            if (content_type or "").split(";")[0].strip().lower() == "text/html":
                raise ValueError("URL returned an HTML page, not a direct media file")

            disposition_name = _content_disposition_filename(response.headers.get("Content-Disposition"))
            filename = _safe_url_filename(disposition_name or response.geturl(), content_type)
            output_path = output_dir / filename
            with open(output_path, "wb") as output_file:
                shutil.copyfileobj(response, output_file)
    except HTTPError as exc:
        raise RuntimeError(f"Media download failed: HTTP {exc.code}") from exc
    except URLError as exc:
        raise RuntimeError(f"Media download failed: {exc.reason}") from exc

    if output_path.suffix.lower() not in MEDIA_EXTENSIONS:
        detected_type, _ = mimetypes.guess_type(output_path.name)
        raise ValueError(f"Unsupported media format: {detected_type or output_path.suffix}")

    return output_path


def _download_with_yt_dlp(url: str, output_dir: Path, direct_error: Exception) -> Path:
    validate_public_media_url(url)
    output_template = str(output_dir / "remote_media.%(ext)s")
    command = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--no-playlist",
        "--no-progress",
        "--restrict-filenames",
        "-f",
        "bestaudio/best",
        "-o",
        output_template,
        url,
    ]

    try:
        subprocess.run(command, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise RuntimeError(f"Media download failed: {direct_error}") from exc

    media_files = [
        path
        for path in output_dir.iterdir()
        if path.is_file() and path.suffix.lower() in MEDIA_EXTENSIONS and path.name.startswith("remote_media.")
    ]
    if not media_files:
        raise RuntimeError("Media download finished but no supported audio/video file was created")

    return max(media_files, key=lambda path: path.stat().st_mtime)


def prepare_audio(input_file: Path, job_dir: Path) -> Path:
    if not input_file.exists():
        raise FileNotFoundError(f"File not found: {input_file}")

    suffix = input_file.suffix.lower()
    if suffix in VIDEO_EXTENSIONS:
        audio_path = job_dir / "work_audio.wav"
        print(f"Extracting audio from video: {input_file}")
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(input_file),
                "-vn",
                "-acodec",
                "pcm_s16le",
                "-ac",
                "1",
                "-ar",
                "16000",
                str(audio_path),
            ],
            check=True,
        )
        return audio_path

    if suffix in AUDIO_EXTENSIONS:
        print(f"Using audio file: {input_file}")
        return input_file

    raise ValueError(f"Unsupported file format: {suffix}")
