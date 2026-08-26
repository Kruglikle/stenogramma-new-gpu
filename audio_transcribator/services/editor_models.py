from typing import Any


EDITOR_MODEL_GROUPS: list[dict[str, Any]] = [
    {
        "label": "Локальные модели",
        "models": [
            {
                "id": "ollama:gemma3:4b",
                "provider": "ollama",
                "model": "gemma3:4b",
                "label": "gemma3:4b",
            },
            {
                "id": "ollama:qwen3:8b",
                "provider": "ollama",
                "model": "qwen3:8b",
                "label": "qwen3:8b",
            },
        ],
    },
]


def list_editor_model_groups() -> list[dict[str, Any]]:
    return EDITOR_MODEL_GROUPS


def resolve_editor_model(model_id: str | None, default_model: str) -> dict[str, Any]:
    requested_model = (model_id or default_model).strip()
    for group in EDITOR_MODEL_GROUPS:
        for model in group["models"]:
            if requested_model in {model["id"], model["model"]}:
                return dict(model)

    raise ValueError(f"Unknown editor model: {requested_model}")
