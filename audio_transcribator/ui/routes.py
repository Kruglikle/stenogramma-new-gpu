import hashlib
import hmac
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from fastapi import APIRouter, Cookie, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from audio_transcribator.auth import verify_credentials
from audio_transcribator.config import settings
from audio_transcribator.services.jobs import (
    StorageQuotaExceeded,
    build_job_result,
    delete_job,
    get_job_file,
    list_user_jobs,
    load_job_metadata,
    start_uploaded_file,
    start_url,
    update_job_title,
    user_storage_quota,
)
from audio_transcribator.services.transcription_models import DEFAULT_TRANSCRIPTION_MODEL_ID, TranscriptionModelError
from audio_transcribator.utils.files import ALLOWED_DOWNLOADS


router = APIRouter(prefix="/ui", include_in_schema=False)
templates = Jinja2Templates(directory=str(settings.base_dir / "audio_transcribator" / "templates"))
BENCHMARK_TYPES = {
    "diarization": {
        "label": "Диаризация",
        "module": "audio_transcribator.benchmarks.diarization",
        "directory": "diarization_benchmarks",
        "summary": "summary.md",
        "aggregate": "aggregate_metrics.json",
        "downloads": {"per_file_metrics.csv", "aggregate_metrics.json", "summary.md", "errors.jsonl", "run.log"},
    },
    "transcription": {
        "label": "Транскрибация",
        "module": "audio_transcribator.benchmarks.transcription",
        "directory": "transcription_benchmarks",
        "summary": "asr_summary.md",
        "aggregate": "asr_aggregate_metrics.json",
        "downloads": {
            "asr_per_file_metrics.csv",
            "asr_aggregate_metrics.json",
            "asr_summary.md",
            "asr_errors.jsonl",
            "run.log",
        },
    },
}
BENCHMARK_PROCESSES: dict[str, subprocess.Popen] = {}


def ui_cookie_options() -> dict:
    return {
        "httponly": True,
        "samesite": "lax",
        "secure": settings.cookie_secure,
        "max_age": settings.ui_session_max_age_seconds,
    }


def sign_ui_user(username: str) -> str:
    return hmac.new(settings.api_token.encode("utf-8"), username.encode("utf-8"), hashlib.sha256).hexdigest()


def require_ui_auth(ui_token: str | None, ui_user: str | None = None, ui_user_sig: str | None = None) -> str:
    if ui_token != settings.api_token or not ui_user or not ui_user_sig:
        raise HTTPException(status_code=status.HTTP_303_SEE_OTHER, headers={"Location": "/ui/login"})

    if not hmac.compare_digest(sign_ui_user(ui_user), ui_user_sig):
        raise HTTPException(status_code=status.HTTP_303_SEE_OTHER, headers={"Location": "/ui/login"})

    return ui_user


def require_safe_ui_post(request: Request) -> None:
    """Reject cross-site UI form posts before cookie-authenticated state changes."""
    host = (request.headers.get("host") or "").lower()
    origin_or_referer = request.headers.get("origin") or request.headers.get("referer")
    if not host or not origin_or_referer:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Missing origin")

    parsed = urlparse(origin_or_referer)
    if (parsed.netloc or "").lower() != host:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid origin")


def build_cabinet_context(username: str) -> dict:
    return {
        "cabinet_jobs": list_user_jobs(username),
        "current_username": username,
        "max_upload_file_bytes": settings.max_upload_file_bytes,
        "storage_quota": user_storage_quota(username),
    }


def can_access_job(username: str, job_id: str) -> bool:
    metadata = load_job_metadata(settings.results_dir / job_id)
    owner = metadata.get("user_login")
    return owner in {None, username}


def parse_optional_positive_int(value: str | int | None) -> int:
    try:
        parsed = int(value or 0)
    except (TypeError, ValueError):
        return 0
    return max(parsed, 0)


def benchmark_config(benchmark_type: str) -> dict:
    return BENCHMARK_TYPES.get(benchmark_type) or BENCHMARK_TYPES["diarization"]


def benchmark_root(benchmark_type: str = "diarization") -> Path:
    root = settings.base_dir / "data" / benchmark_config(benchmark_type)["directory"]
    root.mkdir(parents=True, exist_ok=True)
    return root


def benchmark_run_dir(run_id: str, benchmark_type: str = "diarization") -> Path:
    root = benchmark_root(benchmark_type).resolve()
    run_dir = (root / run_id).resolve()
    if run_dir.parent != root:
        raise HTTPException(status_code=400, detail="Invalid benchmark run id")
    return run_dir


def benchmark_status(run_id: str, run_dir: Path, benchmark_type: str) -> str:
    process = BENCHMARK_PROCESSES.get(run_id)
    if process:
        return_code = process.poll()
        if return_code is None:
            return "running"
        return "completed" if return_code == 0 else "failed"
    if (run_dir / benchmark_config(benchmark_type)["summary"]).exists():
        return "completed"
    if (run_dir / "run.log").exists():
        return "unknown"
    return "missing"


def read_json_file(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def read_tail(path: Path, max_chars: int = 8000) -> str:
    if not path.exists():
        return ""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return text[-max_chars:]


def list_benchmark_runs() -> list[dict]:
    runs = []
    for benchmark_type, config in BENCHMARK_TYPES.items():
        for run_dir in benchmark_root(benchmark_type).iterdir():
            if not run_dir.is_dir():
                continue
            run_id = run_dir.name
            metadata = read_json_file(run_dir / "web_run.json")
            aggregate = read_json_file(run_dir / config["aggregate"])
            files = [name for name in sorted(config["downloads"]) if (run_dir / name).exists()]
            runs.append(
                {
                    "id": run_id,
                    "type": benchmark_type,
                    "type_label": config["label"],
                    "status": benchmark_status(run_id, run_dir, benchmark_type),
                    "created_at": metadata.get("created_at", ""),
                    "command": metadata.get("command", []),
                    "aggregate": aggregate,
                    "files": files,
                    "mtime": run_dir.stat().st_mtime,
                }
            )
    runs.sort(key=lambda item: item["mtime"], reverse=True)
    return runs


def active_benchmark_run() -> str | None:
    for run_id, process in list(BENCHMARK_PROCESSES.items()):
        if process.poll() is None:
            return run_id
    return None


def validate_benchmark_options(limit: int, offset: int, device: str) -> tuple[int, int, str]:
    limit = max(int(limit), 1)
    offset = max(int(offset), 0)
    if device not in {"cpu", "cuda", "auto"}:
        device = "cpu"
    return limit, offset, device


def build_benchmark_context(error: str | None = None, selected_run_id: str | None = None) -> dict:
    runs = list_benchmark_runs()
    selected_run = next((run for run in runs if run["id"] == selected_run_id), runs[0] if runs else None)
    log_tail = (
        read_tail(benchmark_run_dir(selected_run["id"], selected_run["type"]) / "run.log")
        if selected_run
        else ""
    )
    return {
        "error": error,
        "runs": runs[:20],
        "selected_run": selected_run,
        "log_tail": log_tail,
        "active_run_id": active_benchmark_run(),
        "benchmark_types": BENCHMARK_TYPES,
    }


@router.get("", response_class=HTMLResponse)
def ui_root(
    ui_token: str | None = Cookie(default=None),
    ui_user: str | None = Cookie(default=None),
    ui_user_sig: str | None = Cookie(default=None),
):
    try:
        require_ui_auth(ui_token, ui_user, ui_user_sig)
        return RedirectResponse(url="/ui/upload", status_code=status.HTTP_303_SEE_OTHER)
    except HTTPException:
        return RedirectResponse(url="/ui/login", status_code=status.HTTP_303_SEE_OTHER)


@router.get("/login", response_class=HTMLResponse)
def login_page(
    request: Request,
    ui_token: str | None = Cookie(default=None),
    ui_user: str | None = Cookie(default=None),
    ui_user_sig: str | None = Cookie(default=None),
):
    try:
        require_ui_auth(ui_token, ui_user, ui_user_sig)
        return RedirectResponse(url="/ui/upload", status_code=status.HTTP_303_SEE_OTHER)
    except HTTPException:
        pass
    return templates.TemplateResponse(request, "login.html", {"error": None})


@router.post("/login")
def login(request: Request, username: str = Form(...), password: str = Form(...)):
    require_safe_ui_post(request)
    if not verify_credentials(username, password):
        return templates.TemplateResponse(
            request,
            "login.html",
            {"error": "Неверный логин или пароль"},
            status_code=status.HTTP_401_UNAUTHORIZED,
        )

    response = RedirectResponse(url="/ui/upload", status_code=status.HTTP_303_SEE_OTHER)
    response.set_cookie("ui_token", settings.api_token, **ui_cookie_options())
    response.set_cookie("ui_user", username.strip(), **ui_cookie_options())
    response.set_cookie("ui_user_sig", sign_ui_user(username.strip()), **ui_cookie_options())
    return response


@router.post("/logout")
def logout(request: Request):
    require_safe_ui_post(request)
    response = RedirectResponse(url="/ui/login", status_code=status.HTTP_303_SEE_OTHER)
    response.delete_cookie("ui_token", secure=settings.cookie_secure, httponly=True, samesite="lax")
    response.delete_cookie("ui_user", secure=settings.cookie_secure, httponly=True, samesite="lax")
    response.delete_cookie("ui_user_sig", secure=settings.cookie_secure, httponly=True, samesite="lax")
    return response


@router.get("/upload", response_class=HTMLResponse)
def upload_page(
    request: Request,
    ui_token: str | None = Cookie(default=None),
    ui_user: str | None = Cookie(default=None),
    ui_user_sig: str | None = Cookie(default=None),
):
    username = require_ui_auth(ui_token, ui_user, ui_user_sig)
    return templates.TemplateResponse(
        request,
        "upload.html",
        {
            "error": None,
            "diarization_speakers": "",
            **build_cabinet_context(username),
        },
    )


@router.post("/upload")
def upload_file(
    request: Request,
    file: UploadFile | None = File(default=None),
    source_url: str = Form(default=""),
    title: str = Form(default=""),
    enable_summary: bool = Form(default=False),
    diarization_speakers: str = Form(default=""),
    ui_token: str | None = Cookie(default=None),
    ui_user: str | None = Cookie(default=None),
    ui_user_sig: str | None = Cookie(default=None),
):
    require_safe_ui_post(request)
    username = require_ui_auth(ui_token, ui_user, ui_user_sig)
    try:
        parsed_diarization_speakers = parse_optional_positive_int(diarization_speakers)
        clean_source_url = source_url.strip()
        if file and file.filename:
            result = start_uploaded_file(
                file,
                transcription_model_id=DEFAULT_TRANSCRIPTION_MODEL_ID,
                user_login=username,
                title=title,
                enable_transcription=True,
                enable_summary=enable_summary,
                enable_diarization=True,
                diarization_speakers=parsed_diarization_speakers,
            )
        elif clean_source_url:
            result = start_url(
                clean_source_url,
                transcription_model_id=DEFAULT_TRANSCRIPTION_MODEL_ID,
                user_login=username,
                title=title,
                enable_transcription=True,
                enable_summary=enable_summary,
                enable_diarization=True,
                diarization_speakers=parsed_diarization_speakers,
            )
        else:
            raise ValueError("Загрузите файл или вставьте ссылку на медиа")
    except (StorageQuotaExceeded, TranscriptionModelError, ValueError) as exc:
        return templates.TemplateResponse(
            request,
            "upload.html",
            {
                "error": str(exc),
                "diarization_speakers": diarization_speakers,
                **build_cabinet_context(username),
            },
            status_code=status.HTTP_400_BAD_REQUEST,
        )
    return RedirectResponse(url=f"/ui/result/{result['job_id']}", status_code=status.HTTP_303_SEE_OTHER)


@router.get("/benchmark", response_class=HTMLResponse)
def benchmark_page(
    request: Request,
    run_id: str | None = None,
    ui_token: str | None = Cookie(default=None),
    ui_user: str | None = Cookie(default=None),
    ui_user_sig: str | None = Cookie(default=None),
):
    require_ui_auth(ui_token, ui_user, ui_user_sig)
    return templates.TemplateResponse(request, "benchmark.html", build_benchmark_context(selected_run_id=run_id))


@router.post("/benchmark/start")
def start_benchmark(
    request: Request,
    benchmark_type: str = Form(default="diarization"),
    limit: int = Form(default=1),
    offset: int = Form(default=0),
    device: str = Form(default="cpu"),
    save_artifacts: bool = Form(default=False),
    ui_token: str | None = Cookie(default=None),
    ui_user: str | None = Cookie(default=None),
    ui_user_sig: str | None = Cookie(default=None),
):
    require_safe_ui_post(request)
    require_ui_auth(ui_token, ui_user, ui_user_sig)
    active_run_id = active_benchmark_run()
    if active_run_id:
        return templates.TemplateResponse(
            request,
            "benchmark.html",
            build_benchmark_context(
                error=f"Benchmark is already running: {active_run_id}",
                selected_run_id=active_run_id,
            ),
            status_code=status.HTTP_409_CONFLICT,
        )

    benchmark_type = benchmark_type if benchmark_type in BENCHMARK_TYPES else "diarization"
    config = benchmark_config(benchmark_type)
    limit, offset, device = validate_benchmark_options(limit, offset, device)
    run_id = datetime.now(timezone.utc).strftime("web_%Y%m%dT%H%M%S%fZ")
    run_dir = benchmark_run_dir(run_id, benchmark_type)
    run_dir.mkdir(parents=True, exist_ok=False)

    command = [
        sys.executable,
        "-m",
        config["module"],
        "--limit",
        str(limit),
        "--offset",
        str(offset),
        "--device",
        device,
        "--output-dir",
        str(run_dir),
    ]
    if save_artifacts:
        command.append("--save-artifacts")

    metadata = {
        "id": run_id,
        "type": benchmark_type,
        "type_label": config["label"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "limit": limit,
        "offset": offset,
        "device": device,
        "save_artifacts": save_artifacts,
    }
    (run_dir / "web_run.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    with open(run_dir / "run.log", "w", encoding="utf-8", errors="replace") as log_file:
        log_file.write("Command: " + " ".join(command) + "\n\n")
        log_file.flush()
        process = subprocess.Popen(
            command,
            cwd=str(settings.base_dir),
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
    BENCHMARK_PROCESSES[run_id] = process
    return RedirectResponse(url=f"/ui/benchmark?run_id={run_id}", status_code=status.HTTP_303_SEE_OTHER)


@router.get("/benchmark/download/{run_id}/{filename}")
def download_benchmark_file(
    run_id: str,
    filename: str,
    ui_token: str | None = Cookie(default=None),
    ui_user: str | None = Cookie(default=None),
    ui_user_sig: str | None = Cookie(default=None),
):
    require_ui_auth(ui_token, ui_user, ui_user_sig)

    for benchmark_type, config in BENCHMARK_TYPES.items():
        if filename not in config["downloads"]:
            continue
        file_path = benchmark_run_dir(run_id, benchmark_type) / filename
        if file_path.exists() and file_path.is_file():
            return FileResponse(path=str(file_path), filename=filename, media_type="application/octet-stream")
    raise HTTPException(status_code=404, detail="File not found")


@router.get("/result/{job_id}", response_class=HTMLResponse)
def result_page(
    request: Request,
    job_id: str,
    ui_token: str | None = Cookie(default=None),
    ui_user: str | None = Cookie(default=None),
    ui_user_sig: str | None = Cookie(default=None),
):
    username = require_ui_auth(ui_token, ui_user, ui_user_sig)
    try:
        result = build_job_result(job_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Job not found")
    if not can_access_job(username, job_id):
        raise HTTPException(status_code=404, detail="Job not found")

    visible_downloads = {"stenogramma.txt", "diarized_transcript.txt", "summary.txt", "run.log"}
    downloads = [name for name in result["files"] if name in ALLOWED_DOWNLOADS and name in visible_downloads]
    return templates.TemplateResponse(
        request,
        "result.html",
        {
            "result": result,
            "downloads": downloads,
            **build_cabinet_context(username),
        },
    )


@router.post("/result/{job_id}/title")
def rename_result(
    request: Request,
    job_id: str,
    title: str = Form(default=""),
    ui_token: str | None = Cookie(default=None),
    ui_user: str | None = Cookie(default=None),
    ui_user_sig: str | None = Cookie(default=None),
):
    require_safe_ui_post(request)
    username = require_ui_auth(ui_token, ui_user, ui_user_sig)
    if not can_access_job(username, job_id):
        raise HTTPException(status_code=404, detail="Job not found")
    update_job_title(job_id, title)
    return RedirectResponse(url=f"/ui/result/{job_id}", status_code=status.HTTP_303_SEE_OTHER)


@router.post("/result/{job_id}/delete")
def delete_result(
    request: Request,
    job_id: str,
    ui_token: str | None = Cookie(default=None),
    ui_user: str | None = Cookie(default=None),
    ui_user_sig: str | None = Cookie(default=None),
):
    require_safe_ui_post(request)
    username = require_ui_auth(ui_token, ui_user, ui_user_sig)
    if not can_access_job(username, job_id):
        raise HTTPException(status_code=404, detail="Job not found")
    delete_job(job_id)
    return RedirectResponse(url="/ui/upload", status_code=status.HTTP_303_SEE_OTHER)


@router.get("/download/{job_id}/{filename}")
def download_file(
    job_id: str,
    filename: str,
    ui_token: str | None = Cookie(default=None),
    ui_user: str | None = Cookie(default=None),
    ui_user_sig: str | None = Cookie(default=None),
):
    username = require_ui_auth(ui_token, ui_user, ui_user_sig)
    if not can_access_job(username, job_id):
        raise HTTPException(status_code=404, detail="File not found")
    if filename not in ALLOWED_DOWNLOADS:
        raise HTTPException(status_code=403, detail="File is not allowed for download")

    try:
        file_path = get_job_file(job_id, filename)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(path=str(file_path), filename=filename, media_type="application/octet-stream")
