from __future__ import annotations

import json
from importlib.resources import as_file, files
from pathlib import Path

import joblib


def _load_registry() -> dict:
    registry_path = files("turba_models").joinpath("model_registry.json")
    return json.loads(registry_path.read_text(encoding="utf-8"))


def list_models() -> list[dict]:
    """Return all published model entries and their metadata."""
    return _load_registry()["models"]


def _normalize(text: str) -> str:
    return text.strip().lower().replace("_", " ")


def _match_model(model_name: str) -> dict:
    key = _normalize(model_name)
    registry = _load_registry()
    for item in registry["models"]:
        if key in {
            _normalize(item.get("model_key", "")),
            _normalize(item.get("model_name", "")),
            _normalize(item.get("crop_name", "")),
        }:
            return item
    available = ", ".join(m["crop_name"] for m in registry["models"])
    raise ValueError(f"Unknown model: {model_name}. Available models: {available}")


def load_model(model_name: str):
    info = _match_model(model_name)
    resource = files("turba_models").joinpath(info["artifact_path"])
    with as_file(resource) as local_path:
        return joblib.load(Path(local_path))
