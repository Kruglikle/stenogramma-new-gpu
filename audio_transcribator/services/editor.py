import json
from pathlib import Path

from openai import OpenAI

from audio_transcribator.config import settings
from audio_transcribator.services.editor_models import resolve_editor_model
from audio_transcribator.utils.files import write_text_atomic


EDITOR_SYSTEM_PROMPT = """Ты профессиональный русскоязычный редактор стенограмм.

Твоя задача - аккуратно привести фрагмент стенограммы к читаемому виду.
Исправляй орфографию, пунктуацию, очевидные грамматические ошибки и явные сбои распознавания речи.
Расставляй абзацы, убирай повторы и слова-паразиты только там, где это не меняет смысл.
Не добавляй новых фактов. Не сокращай смысл. Не меняй порядок высказываний.
Сохраняй имена, цифры, технические термины, ссылки, таймкоды и названия.
Верни только отредактированный текст без комментариев."""

EDITOR_USER_PROMPT = """Отредактируй этот фрагмент стенограммы.

Фрагмент {part_number} из {parts_count}:
{transcript_part}
"""


def strip_model_thinking(text: str) -> str:
    result = text.strip()
    while "<think>" in result and "</think>" in result:
        start = result.find("<think>")
        end = result.find("</think>", start) + len("</think>")
        result = (result[:start] + result[end:]).strip()
    return result


def split_transcript(text: str, chunk_chars: int) -> list[str]:
    normalized = text.strip()
    if len(normalized) <= chunk_chars:
        return [normalized]

    chunks = []
    current = []
    current_len = 0
    for paragraph in normalized.splitlines():
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        paragraph_len = len(paragraph)
        if current and current_len + paragraph_len + 1 > chunk_chars:
            chunks.append("\n".join(current))
            current = []
            current_len = 0

        if paragraph_len > chunk_chars:
            for start in range(0, paragraph_len, chunk_chars):
                part = paragraph[start : start + chunk_chars].strip()
                if part:
                    chunks.append(part)
            continue

        current.append(paragraph)
        current_len += paragraph_len + 1

    if current:
        chunks.append("\n".join(current))
    return chunks


def build_client(selected_model: dict) -> OpenAI:
    base_url = settings.ollama_base_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"
    return OpenAI(
        api_key=settings.ollama_api_key,
        base_url=base_url,
        timeout=settings.ollama_request_timeout_seconds,
    )


def edit_chunk(client: OpenAI, model: str, chunk: str, part_number: int, parts_count: int) -> str:
    response = client.chat.completions.create(
        model=model,
        temperature=settings.editor_temperature,
        messages=[
            {"role": "system", "content": EDITOR_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": EDITOR_USER_PROMPT.format(
                    part_number=part_number,
                    parts_count=parts_count,
                    transcript_part=chunk,
                ),
            },
        ],
    )
    return strip_model_thinking(response.choices[0].message.content or "")


def edit_transcript(transcript: str, job_dir: Path, model: str | None = None) -> str:
    selected_model = resolve_editor_model(model, settings.editor_model)

    print(f"Editing transcript with {selected_model['model']}...")
    client = build_client(selected_model)
    chunks = split_transcript(transcript, max(settings.editor_chunk_chars, 1500))
    edited_parts = []
    usage_items = []

    for index, chunk in enumerate(chunks, start=1):
        print(f"Editing chunk {index}/{len(chunks)}...", flush=True)
        edited = edit_chunk(client, selected_model["model"], chunk, index, len(chunks))
        if edited:
            edited_parts.append(edited)
            write_text_atomic(job_dir / "edited_transcript.txt", "\n\n".join(edited_parts))
        usage_items.append({"stage": "chunk", "chunk": index})

    result = "\n\n".join(edited_parts)
    write_text_atomic(job_dir / "edited_transcript.txt", result)
    (job_dir / "editing_usage.json").write_text(
        json.dumps(
            {
                "provider": selected_model["provider"],
                "model": selected_model["model"],
                "chunks": len(chunks),
                "usage": usage_items,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Edited transcript saved.")
    return result
