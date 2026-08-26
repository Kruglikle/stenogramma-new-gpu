import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

from audio_transcribator.config import settings
from audio_transcribator.utils.files import write_text_atomic


CHARS_PER_TOKEN_FALLBACK = 4
MIN_SUMMARY_CHUNK_TOKENS = 1000
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+|\n+")

SYSTEM_SUMMARIZE_CHUNK = """Ты - помощник для суммаризации деловых совещаний.
Проанализируй приведенный фрагмент транскрипта и выдай краткую выжимку в виде структурированного текста.
Обязательно укажи:
- Ключевые темы обсуждения
- Принятые решения, если они есть
- Назначенные задачи: что нужно сделать, кто ответственный, срок выполнения

Используй реальные имена или должности из текста. Если имя, ответственный или срок не названы, явно напиши, что они не указаны.
Не придумывай даты, имена и решения. Ответ должен быть на русском языке, без вступлений и воды."""

SYSTEM_MERGE_SUMMARIES = """Ты - профессиональный аналитик. Тебе даны краткие пересказы нескольких частей одного совещания.
Объедини их в единую итоговую сводку, сгруппировав повторяющиеся темы.

Структура отчета:
# Итоги совещания
## Основные темы
## Принятые решения
## План действий

Для задач указывай действие, ответственного и срок. Сроки и ответственных указывай только если они прямо названы в исходных фрагментах.
Не придумывай даты и имена. Ответ дай на русском языке, лаконично и по делу."""

_TOKENIZER: Any | None = None
_TOKENIZER_LOAD_FAILED = False


def resolve_local_tokenizer_model(model: str) -> str:
    configured_path = Path(model).expanduser()
    if configured_path.exists():
        return str(configured_path.resolve())
    if "/" not in model:
        return model

    cache_root = Path(os.environ.get("HF_HUB_CACHE", ""))
    repository_cache = cache_root / f"models--{model.replace('/', '--')}"
    revision = ""
    main_ref = repository_cache / "refs" / "main"
    if main_ref.is_file():
        revision = main_ref.read_text(encoding="utf-8").strip()
    candidates = []
    if revision:
        candidates.append(repository_cache / "snapshots" / revision)
    candidates.extend(sorted((repository_cache / "snapshots").glob("*"), reverse=True))
    for candidate in candidates:
        if (candidate / "config.json").is_file():
            return str(candidate.resolve())
    return model


def get_tokenizer() -> Any | None:
    global _TOKENIZER, _TOKENIZER_LOAD_FAILED
    if _TOKENIZER is not None:
        return _TOKENIZER
    if _TOKENIZER_LOAD_FAILED:
        return None

    try:
        from transformers import AutoTokenizer

        tokenizer_model = resolve_local_tokenizer_model(settings.summary_tokenizer_model)
        _TOKENIZER = AutoTokenizer.from_pretrained(
            tokenizer_model,
            trust_remote_code=True,
            local_files_only=settings.summary_local_files_only,
            use_fast=False,
        )
        return _TOKENIZER
    except Exception as exc:
        _TOKENIZER_LOAD_FAILED = True
        print(f"Summary tokenizer is unavailable, using approximate token count: {exc}", flush=True)
        return None


def count_tokens(text: str) -> int:
    tokenizer = get_tokenizer()
    if tokenizer is None:
        return max(1, math.ceil(len(text) / CHARS_PER_TOKEN_FALLBACK))
    return len(tokenizer.encode(text))


def split_oversized_unit(text: str, max_tokens: int) -> list[str]:
    tokenizer = get_tokenizer()
    if tokenizer is None:
        max_chars = max(max_tokens * CHARS_PER_TOKEN_FALLBACK, 1)
        return [text[start : start + max_chars].strip() for start in range(0, len(text), max_chars) if text[start : start + max_chars].strip()]

    tokens = tokenizer.encode(text)
    chunks = []
    for start in range(0, len(tokens), max_tokens):
        part = tokenizer.decode(tokens[start : start + max_tokens]).strip()
        if part:
            chunks.append(part)
    return chunks


def strip_model_thinking(text: str) -> str:
    result = text.strip()
    while "<think>" in result and "</think>" in result:
        start = result.find("<think>")
        end = result.find("</think>", start) + len("</think>")
        result = (result[:start] + result[end:]).strip()
    return re.sub(r"<think>.*", "", result, flags=re.DOTALL).strip()


def chunk_text_with_overlap(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    normalized = text.strip()
    if not normalized:
        return []

    max_tokens = max(max_tokens, MIN_SUMMARY_CHUNK_TOKENS)
    overlap_tokens = max(0, min(overlap_tokens, max_tokens // 4))
    units = [unit.strip() for unit in SENTENCE_SPLIT_RE.split(normalized) if unit.strip()]

    chunks: list[str] = []
    current_units: list[str] = []
    current_tokens = 0

    for unit in units:
        unit_parts = [unit]
        unit_tokens = count_tokens(unit)
        if unit_tokens > max_tokens:
            unit_parts = split_oversized_unit(unit, max_tokens)

        for part in unit_parts:
            part_tokens = count_tokens(part)
            if current_units and current_tokens + part_tokens > max_tokens:
                chunks.append(" ".join(current_units).strip())
                overlap_units: list[str] = []
                overlap_total = 0
                for previous_unit in reversed(current_units):
                    previous_tokens = count_tokens(previous_unit)
                    if overlap_total + previous_tokens > overlap_tokens:
                        break
                    overlap_units.insert(0, previous_unit)
                    overlap_total += previous_tokens
                current_units = overlap_units
                current_tokens = overlap_total

            current_units.append(part)
            current_tokens += part_tokens

    if current_units:
        chunks.append(" ".join(current_units).strip())

    return chunks


def build_vllm_client() -> OpenAI:
    base_url = settings.vllm_base_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"
    return OpenAI(
        api_key=settings.vllm_api_key,
        base_url=base_url,
        timeout=settings.summary_request_timeout_seconds,
    )


def call_summary_model(
    client: OpenAI,
    messages: list[dict[str, str]],
    system_prompt: str,
    max_tokens: int,
    temperature: float = 0.3,
) -> tuple[str, dict[str, Any]]:
    last_error: Exception | None = None
    for attempt in range(settings.summary_request_retries + 1):
        try:
            response = client.chat.completions.create(
                model=settings.summary_model,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=[{"role": "system", "content": system_prompt}, *messages],
                extra_body={
                    "chat_template_kwargs": {
                        "enable_thinking": settings.summary_enable_thinking,
                    }
                },
            )
            content = response.choices[0].message.content or ""
            usage = response.usage.model_dump() if response.usage else {}
            content = strip_model_thinking(content)
            if not content:
                raise RuntimeError("Summary model returned an empty response")
            return content, usage
        except Exception as exc:
            last_error = exc
            if attempt >= settings.summary_request_retries:
                break
            delay_seconds = 2**attempt
            print(f"Summary request failed: {exc}. Retrying in {delay_seconds}s...", flush=True)
            time.sleep(delay_seconds)

    raise RuntimeError(f"Could not call vLLM summary model after retries: {last_error}")


def write_summary_usage(job_dir: Path, usage_items: list[dict[str, Any]]) -> None:
    if not usage_items:
        return

    (job_dir / "summary_usage.json").write_text(
        json.dumps(
            {
                "provider": "vllm",
                "base_url": settings.vllm_base_url,
                "model": settings.summary_model,
                "usage": usage_items,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def reduce_partial_summaries(
    client: OpenAI,
    partial_summaries: list[str],
    usage_items: list[dict[str, Any]],
) -> str:
    combined = "\n\n---\n\n".join(partial_summaries)
    level = 1
    while count_tokens(combined) > settings.summary_max_chunk_tokens:
        groups = chunk_text_with_overlap(combined, settings.summary_max_chunk_tokens, 0)
        reduced: list[str] = []
        for index, group in enumerate(groups, start=1):
            print(f"Reducing summaries, level {level}, group {index}/{len(groups)}...", flush=True)
            merged, usage = call_summary_model(
                client,
                messages=[{"role": "user", "content": group}],
                system_prompt=SYSTEM_MERGE_SUMMARIES,
                max_tokens=settings.summary_chunk_max_tokens,
            )
            reduced.append(f"### Merge {level}.{index}\n{merged}")
            usage_items.append(
                {
                    "stage": "reduce",
                    "level": level,
                    "group": index,
                    "groups": len(groups),
                    "usage": usage,
                }
            )
        next_combined = "\n\n---\n\n".join(reduced)
        if count_tokens(next_combined) >= count_tokens(combined):
            raise RuntimeError("Summary reduction did not reduce the token count")
        combined = next_combined
        level += 1
    return combined


def summarize(transcript: str, job_dir: Path, progress_callback=None) -> str | None:
    if not transcript.strip():
        print("Transcript is empty, skipping summary")
        return None

    print(f"Summarizing with vLLM model {settings.summary_model}...", flush=True)
    client = build_vllm_client()
    total_tokens = count_tokens(transcript)
    usage_items: list[dict[str, Any]] = []

    if total_tokens <= settings.summary_max_chunk_tokens:
        result, usage = call_summary_model(
            client,
            messages=[{"role": "user", "content": transcript}],
            system_prompt=SYSTEM_SUMMARIZE_CHUNK,
            max_tokens=settings.summary_final_max_tokens,
        )
        write_text_atomic(job_dir / "summary.txt", result)
        usage_items.append({"stage": "single", "input_tokens_estimate": total_tokens, "usage": usage})
        write_summary_usage(job_dir, usage_items)
        if progress_callback:
            progress_callback(100)
        return result

    chunks = chunk_text_with_overlap(
        transcript,
        settings.summary_max_chunk_tokens,
        settings.summary_overlap_tokens,
    )
    partial_summaries: list[str] = []

    for index, chunk in enumerate(chunks, start=1):
        print(f"Summarizing chunk {index}/{len(chunks)}...", flush=True)
        partial, usage = call_summary_model(
            client,
            messages=[{"role": "user", "content": chunk}],
            system_prompt=SYSTEM_SUMMARIZE_CHUNK,
            max_tokens=settings.summary_chunk_max_tokens,
        )
        if partial:
            partial_summaries.append(f"### Часть {index}\n{partial}")
            write_text_atomic(
                job_dir / "summary.txt",
                f"Резюме готовится. Уже обработано частей: {index}/{len(chunks)}.\n\n"
                + "\n\n".join(partial_summaries),
            )
        usage_items.append(
            {
                "stage": "chunk",
                "chunk": index,
                "input_tokens_estimate": count_tokens(chunk),
                "usage": usage,
            }
        )
        if progress_callback:
            progress_callback(index / (len(chunks) + 1) * 100, f"Обработано частей: {index}/{len(chunks)}")

    combined = reduce_partial_summaries(client, partial_summaries, usage_items)
    if progress_callback:
        progress_callback(len(chunks) / (len(chunks) + 1) * 100, "Финальная сборка резюме")

    final_summary, usage = call_summary_model(
        client,
        messages=[{"role": "user", "content": f"Фрагменты суммаризаций:\n{combined}"}],
        system_prompt=SYSTEM_MERGE_SUMMARIES,
        max_tokens=settings.summary_final_max_tokens,
    )
    write_text_atomic(job_dir / "summary.txt", final_summary)
    usage_items.append({"stage": "final", "chunks": len(chunks), "usage": usage})
    write_summary_usage(job_dir, usage_items)

    print("Summary saved.", flush=True)
    if progress_callback:
        progress_callback(100)
    return final_summary
