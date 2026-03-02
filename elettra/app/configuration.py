#!/usr/bin/env python3

"""Dataclass-based configuration loader and UI defaults for Streamlit apps."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import streamlit as st
import yaml

CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


@dataclass(slots=True)
class GlobalConfig:
    """Global settings shared by all apps."""

    prefix: str = "BPM"
    control_system: str = "epics"
    device: str = "cpu"
    dtype: str = "float64"


@dataclass(slots=True)
class ScriptConfig:
    """App-specific defaults."""

    values: dict[str, Any] = field(default_factory=dict)
    x: dict[str, Any] = field(default_factory=dict)
    y: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AppConfig:
    """Typed app configuration."""

    global_: GlobalConfig
    script: ScriptConfig


def _normalize_label(text: str) -> str:
    token = re.sub(r"[^0-9a-zA-Z]+", "_", text.strip().lower())
    token = re.sub(r"_+", "_", token).strip("_")
    return token


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    return {}


def _parse_plane_and_key(label: str) -> tuple[str | None, str]:
    token = label.strip()
    match = re.search(r"\(([^)]+)\)\s*$", token)
    plane = None
    if match is not None:
        suffix = match.group(1).strip().lower()
        if suffix.startswith("x"):
            plane = "x"
        elif suffix.startswith("y"):
            plane = "y"
        token = token[: match.start()].strip()
    return plane, _normalize_label(token)


def _build_global(payload: dict[str, Any]) -> GlobalConfig:
    raw = _as_dict(payload.get("global"))
    legacy = {key: payload.get(key) for key in ("prefix", "control_system", "device", "dtype") if key in payload}
    merged = {
        "prefix": "BPM",
        "control_system": "epics",
        "device": "cpu",
        "dtype": "float64",
        **{key: str(value) for key, value in legacy.items()},
        **{key: str(value) for key, value in raw.items()},
    }

    if merged["control_system"] not in {"epics", "tango"}:
        raise RuntimeError("config.global.control_system must be 'epics' or 'tango'.")
    if merged["device"] not in {"cpu", "cuda"}:
        raise RuntimeError("config.global.device must be 'cpu' or 'cuda'.")
    if merged["dtype"] not in {"float32", "float64"}:
        raise RuntimeError("config.global.dtype must be 'float32' or 'float64'.")

    return GlobalConfig(**merged)


def _build_script(payload: dict[str, Any], script_name: str) -> ScriptConfig:
    raw = _as_dict(payload.get(script_name))
    return ScriptConfig(
        values={key: value for key, value in raw.items() if key not in {"x", "y"}},
        x=_as_dict(raw.get("x")),
        y=_as_dict(raw.get("y")),
    )


def load_app_config(script_name: str) -> AppConfig:
    """Load global + app defaults from YAML into dataclasses."""

    if not CONFIG_PATH.exists():
        raise RuntimeError(f"Config file not found: {CONFIG_PATH}")

    with CONFIG_PATH.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream) or {}

    if not isinstance(payload, dict):
        raise RuntimeError("Config root must be a mapping.")

    global_cfg = _build_global(payload)
    script_cfg = _build_script(payload, script_name)
    return AppConfig(global_=global_cfg, script=script_cfg)


@dataclass(slots=True)
class WidgetDefaults:
    """Adapter around Streamlit widgets that injects config defaults."""

    script: ScriptConfig

    def _lookup(self, label: str, fallback: Any) -> Any:
        plane, key = _parse_plane_and_key(label)
        if plane == "x" and key in self.script.x:
            return self.script.x[key]
        if plane == "y" and key in self.script.y:
            return self.script.y[key]
        if key in self.script.values:
            return self.script.values[key]
        return fallback

    @staticmethod
    def _cast_like(value: Any, fallback: Any) -> Any:
        if isinstance(fallback, bool):
            return bool(value)
        if isinstance(fallback, int):
            try:
                return int(value)
            except (TypeError, ValueError):
                return fallback
        if isinstance(fallback, float):
            try:
                return float(value)
            except (TypeError, ValueError):
                return fallback
        if isinstance(fallback, str):
            return str(value)
        return value

    def number_input(self, label: str, *args: Any, **kwargs: Any) -> Any:
        if "value" in kwargs:
            fallback = kwargs["value"]
            configured = self._lookup(label, fallback)
            kwargs["value"] = self._cast_like(configured, fallback)
        return st.number_input(label, *args, **kwargs)

    def checkbox(self, label: str, *args: Any, **kwargs: Any) -> Any:
        if "value" in kwargs:
            fallback = kwargs["value"]
            configured = self._lookup(label, fallback)
            kwargs["value"] = bool(configured)
        return st.checkbox(label, *args, **kwargs)

    def text_input(self, label: str, *args: Any, **kwargs: Any) -> Any:
        if "value" in kwargs:
            fallback = kwargs["value"]
            configured = self._lookup(label, fallback)
            kwargs["value"] = str(configured)
        return st.text_input(label, *args, **kwargs)

    def selectbox(self, label: str, options: Any, *args: Any, **kwargs: Any) -> Any:
        options_list = list(options)
        if options_list:
            index = int(kwargs.get("index", 0))
            index = max(0, min(index, len(options_list) - 1))
            fallback = options_list[index]
            configured = self._lookup(label, fallback)
            if configured in options_list:
                kwargs["index"] = options_list.index(configured)
        return st.selectbox(label, options, *args, **kwargs)

    def radio(self, label: str, options: Any, *args: Any, **kwargs: Any) -> Any:
        options_list = list(options)
        if options_list:
            index = int(kwargs.get("index", 0))
            index = max(0, min(index, len(options_list) - 1))
            fallback = options_list[index]
            configured = self._lookup(label, fallback)
            if configured in options_list:
                kwargs["index"] = options_list.index(configured)
        return st.radio(label, options, *args, **kwargs)
