#!/usr/bin/env python3
"""
Скрипт для суммаризации длительного текста совещания (транскрипта) через vLLM (Qwen3-8B-AWQ).
Реализует иерархический подход: разбивает на шарды с перекрытием, суммаризирует каждый,
затем объединяет результаты в итоговую сводку.
"""

import requests
import json
from pathlib import Path
import re
import time
from typing import List, Optional

# Попытка импортировать токенизатор Qwen для точного подсчёта
try:
    from transformers import AutoTokenizer
    TOKENIZER = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B-AWQ", trust_remote_code=True)
    def count_tokens(text: str) -> int:
        return len(TOKENIZER.encode(text))
    print("[+] Используется точный токенизатор Qwen")
except ImportError:
    # Запасной вариант – tiktoken (приблизительно)
    import tiktoken
    try:
        enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    def count_tokens(text: str) -> int:
        return len(enc.encode(text))
    print("[!] Transformers не установлен, используется приблизительный подсчёт (tiktoken). Рекомендуется установить transformers: pip install transformers")

# ====================== НАСТРОЙКИ ======================
VLLM_URL = "http://localhost:8000/v1/chat/completions"
MODEL_NAME = "Qwen/Qwen3-8B-AWQ"

INPUT_FILE = "совещание.txt"          # файл с транскриптом
OUTPUT_FILE = "meeting_summary.txt"   # файл для итоговой сводки

# Лимиты (учитываем, что vLLM запущен с --max-model-len 24000)
MAX_MODEL_LEN = 22000                 # оставляем запас относительно 24000
MAX_CHUNK_TOKENS = 18000              # максимальный размер ВХОДА для одного шарда
OVERLAP_TOKENS = 200                  # перекрытие между шардами (токенов)
MAX_SUMMARY_TOKENS = 1200             # длина суммаризации ОДНОГО шарда
FINAL_SUMMARY_TOKENS = 2500           # длина ИТОГОВОЙ сводки

# Убедимся, что сумма не превышает лимит модели
assert MAX_CHUNK_TOKENS + MAX_SUMMARY_TOKENS <= MAX_MODEL_LEN, "Превышен максимальный контекст модели!"

# Промпты (улучшенные, структурированные)
SYSTEM_SUMMARIZE_CHUNK = """Ты — помощник для суммаризации деловых совещаний. 
Проанализируй приведённый фрагмент транскрипта и выдай краткую выжимку в виде структурированного текста. 
Обязательно укажи:
- Ключевые темы обсуждения
- Принятые решения (если есть)
- Назначенные задачи – для каждой задачи опиши:
   * что именно нужно сделать,
   * кто ответственный (используй реальное имя или должность из текста; избегай обобщений типа "Спикер N" или "команда", если есть конкретное имя),
   * срок выполнения (если указан в тексте, иначе напиши "срок не указан").
Ответ должен быть на русском языке, без вступлений и воды."""

SYSTEM_MERGE_SUMMARIES = """Ты — профессиональный аналитик. Тебе даны краткие пересказы нескольких частей одного совещания.
Объедини их в единую итоговую сводку, сгруппировав повторяющиеся темы. 
Структура отчёта:
# Итоги совещания
## Основные темы
## Принятые решения
## План действий (задачи, сроки, ответственные)
**Важно:** 
- Сроки указывай только если они прямо названы в исходных фрагментах. Если срок не упомянут – пиши "срок не определён".
- Ответственных указывай по реальным именам/должностям из текста. Если имя не удаётся определить – пиши "ответственный не указан".
Не придумывай даты и имена.
Ответ дай на русском языке, лаконично и по делу."""

# ========================================================

def clean_response(text: str) -> str:
    """Удаляет возможные теги <think> (на случай, если модель их выдаст)."""
    # Ищем закрывающий тег </think> и берём всё после него
    match = re.search(r'</think>\s*(.*)', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Если есть открывающий без закрывающего – удаляем всё до конца
    text = re.sub(r'<think>.*', '', text, flags=re.DOTALL).strip()
    return text

def call_vllm(messages: List[dict], system_prompt: Optional[str] = None,
              max_tokens: int = MAX_SUMMARY_TOKENS, temperature: float = 0.3,
              retries: int = 2) -> str:
    """
    Вызывает vLLM API с заданными параметрами.
    При ошибке выполняет повторные попытки.
    """
    payload = {
        "model": MODEL_NAME,
        "messages": ([{"role": "system", "content": system_prompt}] if system_prompt else []) + messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    headers = {"Content-Type": "application/json"}

    for attempt in range(retries + 1):
        try:
            resp = requests.post(VLLM_URL, json=payload, headers=headers, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            raw_content = data["choices"][0]["message"]["content"]
            return clean_response(raw_content)
        except Exception as e:
            if attempt < retries:
                wait = 2 ** attempt
                print(f"⚠️  Ошибка: {e}. Повтор через {wait} сек...")
                time.sleep(wait)
            else:
                raise RuntimeError(f"Не удалось вызвать vLLM после {retries+1} попыток: {e}")

def chunk_text_with_overlap(text: str, max_tokens: int, overlap_tokens: int) -> List[str]:
    """
    Разбивает текст на шарды с перекрытием.
    Перекрытие реализовано путём сохранения последних overlap_tokens из предыдущего шарда.
    Разбиение происходит по предложениям (по точке с пробелом), чтобы не резать слова.
    """
    # Разобьём на предложения (примитивно, но для большинства случаев работает)
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s for s in sentences if s]

    chunks = []
    current_chunk = []
    current_len = 0

    for sent in sentences:
        sent_tokens = count_tokens(sent)
        # Если одно предложение превышает лимит – принудительно обрежем его (редкий случай)
        if sent_tokens > max_tokens:
            # Обрезаем по символам (грубо) – можно улучшить
            while count_tokens(sent) > max_tokens:
                sent = sent[:int(len(sent)*0.9)]
            sent_tokens = count_tokens(sent)

        # Если добавление предложения превысит лимит, сохраняем текущий чанк и начинаем новый с перекрытием
        if current_len + sent_tokens > max_tokens:
            # Формируем чанк из текущего набора предложений
            chunk_text = " ".join(current_chunk)
            chunks.append(chunk_text)

            # Перекрытие: берём последние предложения (по токенам) из текущего чанка
            overlap_chunk = []
            overlap_len = 0
            for s in reversed(current_chunk):
                s_tokens = count_tokens(s)
                if overlap_len + s_tokens <= overlap_tokens:
                    overlap_chunk.insert(0, s)
                    overlap_len += s_tokens
                else:
                    break
            # Новый чанк начинаем с перекрывающихся предложений
            current_chunk = overlap_chunk
            current_len = overlap_len

        current_chunk.append(sent)
        current_len += sent_tokens

    # Последний чанк
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def main():
    # 1. Загрузка текста
    if not Path(INPUT_FILE).exists():
        print(f"❌ Файл {INPUT_FILE} не найден!")
        return

    text = Path(INPUT_FILE).read_text(encoding="utf-8").strip()
    print(f"[+] Исходный размер: {len(text)} символов")
    total_tokens = count_tokens(text)
    print(f"[+] Оценка токенов: ~{total_tokens}")

    # 2. Если текст короткий – обрабатываем целиком
    if total_tokens <= MAX_CHUNK_TOKENS:
        print("[+] Текст помещается в один шард – обрабатываем напрямую.")
        summary = call_vllm(
            messages=[{"role": "user", "content": text}],
            system_prompt=SYSTEM_SUMMARIZE_CHUNK,
            max_tokens=FINAL_SUMMARY_TOKENS   # можно сразу дать финальный объём
        )
        print("\n=== ИТОГОВАЯ СУММАРИЗАЦИЯ ===\n" + summary)
        Path(OUTPUT_FILE).write_text(summary, encoding="utf-8")
        print(f"[+] Результат сохранён в {OUTPUT_FILE}")
        return

    # 3. Разбиваем на шарды с перекрытием
    print(f"[!] Текст длинный – разбиваем на шарды (макс. {MAX_CHUNK_TOKENS} токенов, перекрытие {OVERLAP_TOKENS} токенов)...")
    chunks = chunk_text_with_overlap(text, MAX_CHUNK_TOKENS, OVERLAP_TOKENS)
    print(f"[+] Получено {len(chunks)} шардов")

    # 4. Обработка каждого шарда
    summaries = []
    for i, chunk in enumerate(chunks, 1):
        chunk_tokens = count_tokens(chunk)
        print(f"[{i}/{len(chunks)}] Обработка шарда (токенов: {chunk_tokens})...")
        summary = call_vllm(
            messages=[{"role": "user", "content": chunk}],
            system_prompt=SYSTEM_SUMMARIZE_CHUNK,
            max_tokens=MAX_SUMMARY_TOKENS
        )
        summaries.append(summary)
        print(f"    → Суммаризация: {count_tokens(summary)} токенов")

    # 5. Финальное объединение
    print("\n[+] Генерация итоговой сводки из фрагментов...")
    combined = "\n\n---\n\n".join(summaries)
    final_summary = call_vllm(
        messages=[{"role": "user", "content": f"Фрагменты суммаризаций:\n{combined}"}],
        system_prompt=SYSTEM_MERGE_SUMMARIES,
        max_tokens=FINAL_SUMMARY_TOKENS
    )

    # 6. Сохранение и вывод
    Path(OUTPUT_FILE).write_text(final_summary, encoding="utf-8")
    print("\n=== ИТОГОВАЯ СУММАРИЗАЦИЯ ===\n" + final_summary)
    print(f"\n[+] Результат сохранён в {OUTPUT_FILE}")

if __name__ == "__main__":
    main()