"""Settings: four model roles, each an OpenAI-compatible endpoint, plus the knobs around them.

    ASR         speech -> text           POST {base}/audio/transcriptions
    TTS         text -> speech           POST {base}/audio/speech
    FOREGROUND  holds the conversation   POST {base}/chat/completions   (also judges end-of-turn)
    BACKGROUND  works on handed-off tasks POST {base}/chat/completions

Every role reads <ROLE>_BASE_URL, <ROLE>_API_KEY and <ROLE>_MODEL. ASR and TTS run in-process when their
base URL is blank or "local"; BACKGROUND falls back to FOREGROUND field by field; FOREGROUND falls back to
OPENAI_BASE_URL / OPENAI_API_KEY. VAD is not a role: it runs on every 32 ms frame, in-process.

One more role is optional, and replaces ASR, the foreground model and TTS when the dashboard switches to it:

    DUPLEX      a full-duplex speech model  ws {base}/realtime      (see duplex.py)

Precedence, highest first: command-line flags, environment, .env in the working directory, the TOML file
(--config, or ./interaction-harness.toml when it exists).
"""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from dotenv import dotenv_values

from . import prompts

THINKING_MODES = ("chat_template_kwargs", "reasoning_effort", "off")
LOCAL_BACKENDS = ("auto", "mlx", "onnx")
DEFAULT_TOML = "interaction-harness.toml"
NO_KEY = "not-needed"


@dataclass
class Endpoint:
    """One model role. `base_url` None means in-process (ASR/TTS only)."""

    role: str
    base_url: str | None
    api_key: str
    model: str

    @property
    def local(self) -> bool:
        return self.base_url is None

    @property
    def host(self) -> str:
        """Where the role runs, without path or credentials, for status lines and the dashboard."""
        if self.base_url is None:
            return "in-process"
        p = urlparse(self.base_url)
        return p.hostname + (f":{p.port}" if p.port else "") if p.hostname else self.base_url


@dataclass
class Settings:
    foreground: Endpoint
    background: Endpoint
    asr: Endpoint
    tts: Endpoint
    duplex: Endpoint = field(default_factory=lambda: Endpoint("duplex", None, NO_KEY, ""))  # base_url None: off
    foreground_thinking: str = "chat_template_kwargs"
    background_thinking: str = "chat_template_kwargs"
    background_temperature: float = 0.7
    background_rounds: int = 10          # model calls per background task
    asr_language: str | None = None
    asr_partial_ms: int = 640            # partial transcript cadence; 0 = only when the judge needs one
    tts_voice: str | None = None         # None: the backend's first voice
    tts_voices: list[str] = field(default_factory=list)
    tts_format: str = "wav"              # endpoint TTS: wav or pcm
    tts_sample_rate: int = 24_000        # endpoint TTS with pcm: the server's rate
    tts_speed: float = 1.05
    backend: str = "auto"                # in-process speech: auto | mlx | onnx
    device: str = "cpu"                  # onnx backend: cpu | cuda
    parakeet_onnx_variant: str = "int8"  # fp32 | int8 (same transcripts in our tests; a quarter the download)
    kokoro_onnx_variant: str = "int8"    # fp32 | fp16 | int8 (about twice as fast on CPU)
    foreground_prompt: str = prompts.FOREGROUND_SYSTEM
    background_prompt: str = prompts.BACKGROUND_SYSTEM
    end_of_turn_prompt: str = prompts.END_OF_TURN_SYSTEM
    duplex_prompt: str = prompts.DUPLEX_SYSTEM
    tools_files: list[str] = field(default_factory=list)
    save_utterances: str | None = None   # directory: keep a WAV of every user turn the ASR heard
    host: str = "127.0.0.1"
    port: int = 8765
    ssl_certfile: str | None = None
    ssl_keyfile: str | None = None

    def roles(self) -> dict[str, Endpoint]:
        roles = {"asr": self.asr, "tts": self.tts, "foreground": self.foreground, "background": self.background}
        if self.duplex.base_url:
            roles["duplex"] = self.duplex
        return roles


def _same_host(a: str, b: str) -> bool:
    pa, pb = urlparse(a), urlparse(b)
    return bool(pa.hostname) and (pa.scheme, pa.hostname, pa.port) == (pb.scheme, pb.hostname, pb.port)


def _flatten_toml(data: dict[str, Any]) -> dict[str, str]:
    """[foreground] base_url = ... -> FOREGROUND_BASE_URL; [prompts] foreground = ... -> FOREGROUND_PROMPT;
    [server] port = ... (or a top-level port = ...) -> PORT."""
    out: dict[str, str] = {}
    for key, value in data.items():
        if isinstance(value, dict):
            for k, v in value.items():
                name = f"{k}_PROMPT" if key == "prompts" else k if key == "server" else f"{key}_{k}"
                out[name.upper()] = ",".join(map(str, v)) if isinstance(v, list) else str(v)
        else:
            out[key.upper()] = ",".join(map(str, value)) if isinstance(value, list) else str(value)
    return out


def _sources(flags: dict[str, Any] | None, config_path: str | None, env: dict[str, str] | None,
             dotenv_path: str | None) -> dict[str, str]:
    merged: dict[str, str] = {}
    toml_path = Path(config_path) if config_path else Path(DEFAULT_TOML)
    if config_path and not toml_path.exists():
        raise FileNotFoundError(f"config file not found: {toml_path}")
    if toml_path.exists():
        merged.update(_flatten_toml(tomllib.loads(toml_path.read_text())))
    dotenv = Path(dotenv_path) if dotenv_path else Path(".env")
    if dotenv.exists():
        merged.update({k: v for k, v in dotenv_values(dotenv).items() if v is not None})
    merged.update(os.environ if env is None else env)
    for k, v in (flags or {}).items():
        if v is not None:
            merged[k.upper()] = ",".join(map(str, v)) if isinstance(v, list) else str(v)
    return merged


def load_settings(flags: dict[str, Any] | None = None, *, config_path: str | None = None,
                  env: dict[str, str] | None = None, dotenv_path: str | None = None,
                  require_foreground: bool = True) -> Settings:
    """Build Settings from flags, env, .env and TOML. `flags` uses env-style keys (FOREGROUND_MODEL=...)."""
    src = _sources(flags, config_path, env, dotenv_path)

    def get(*keys: str, default: str | None = None) -> str | None:
        for k in keys:
            v = src.get(k)
            if v is not None and str(v).strip() != "":
                return str(v).strip()
        return default

    def speech_url(role: str) -> str | None:
        url = get(f"{role}_BASE_URL")
        return None if url is None or url.lower() == "local" else url.rstrip("/")

    fg_url = (get("FOREGROUND_BASE_URL", "OPENAI_BASE_URL") or "").rstrip("/")
    fg_key = get("FOREGROUND_API_KEY", "OPENAI_API_KEY", default=NO_KEY)

    def key_for(role: str, url: str | None) -> str:
        # a key only goes to the host it was set for: reuse the foreground key on the same host, never elsewhere
        own = get(f"{role}_API_KEY")
        if own:
            return own
        return fg_key if url and _same_host(url, fg_url) else NO_KEY

    foreground = Endpoint("foreground", fg_url, fg_key, get("FOREGROUND_MODEL", default=""))
    bg_url = (get("BACKGROUND_BASE_URL") or fg_url).rstrip("/")
    background = Endpoint("background", bg_url, key_for("BACKGROUND", bg_url),
                          get("BACKGROUND_MODEL", default="") or (foreground.model if bg_url == fg_url else ""))
    asr_url, tts_url = speech_url("ASR"), speech_url("TTS")
    asr = Endpoint("asr", asr_url, key_for("ASR", asr_url), get("ASR_MODEL", default=""))
    tts = Endpoint("tts", tts_url, key_for("TTS", tts_url), get("TTS_MODEL", default=""))

    dx_url = get("DUPLEX_BASE_URL")
    dx_url = None if dx_url is None or dx_url.lower() == "off" else dx_url.rstrip("/")
    duplex = Endpoint("duplex", dx_url, key_for("DUPLEX", dx_url), get("DUPLEX_MODEL", default=""))

    s = Settings(foreground=foreground, background=background, asr=asr, tts=tts, duplex=duplex)
    s.foreground_thinking = get("FOREGROUND_THINKING", default=s.foreground_thinking)
    # same server as the foreground: same way of switching thinking, unless told otherwise
    s.background_thinking = get("BACKGROUND_THINKING", default=s.foreground_thinking if bg_url == fg_url
                                else s.background_thinking)
    s.background_temperature = float(get("BACKGROUND_TEMPERATURE", default=str(s.background_temperature)))
    s.background_rounds = int(get("BACKGROUND_ROUNDS", default=str(s.background_rounds)))
    s.asr_language = get("ASR_LANGUAGE")
    s.asr_partial_ms = int(get("ASR_PARTIAL_MS", default=str(s.asr_partial_ms)))
    s.tts_voice = get("TTS_VOICE")
    s.tts_voices = [v.strip() for v in (get("TTS_VOICES") or "").split(",") if v.strip()]
    s.tts_format = get("TTS_FORMAT", default=s.tts_format).lower()
    s.tts_sample_rate = int(get("TTS_SAMPLE_RATE", default=str(s.tts_sample_rate)))
    s.tts_speed = float(get("TTS_SPEED", default=str(s.tts_speed)))
    s.backend = get("BACKEND", default=s.backend).lower()
    s.device = get("DEVICE", default=s.device).lower()
    s.parakeet_onnx_variant = get("PARAKEET_ONNX_VARIANT", default=s.parakeet_onnx_variant).lower()
    s.kokoro_onnx_variant = get("KOKORO_ONNX_VARIANT", default=s.kokoro_onnx_variant).lower()
    s.foreground_prompt = get("FOREGROUND_PROMPT", default=s.foreground_prompt)
    s.background_prompt = get("BACKGROUND_PROMPT", default=s.background_prompt)
    s.end_of_turn_prompt = get("END_OF_TURN_PROMPT", default=s.end_of_turn_prompt)
    s.duplex_prompt = get("DUPLEX_PROMPT", default=s.duplex_prompt)
    s.tools_files = [p.strip() for p in (get("TOOLS") or "").split(",") if p.strip()]
    s.save_utterances = get("SAVE_UTTERANCES")
    s.host = get("HOST", default=s.host)
    s.port = int(get("PORT", default=str(s.port)))
    s.ssl_certfile = get("SSL_CERTFILE")
    s.ssl_keyfile = get("SSL_KEYFILE")
    validate(s, require_foreground)
    return s


def validate(s: Settings, require_foreground: bool = True) -> None:
    for name in ("foreground_thinking", "background_thinking"):
        if getattr(s, name) not in THINKING_MODES:
            raise ValueError(f"{name.upper()} must be one of {', '.join(THINKING_MODES)}; got {getattr(s, name)!r}")
    if s.backend not in LOCAL_BACKENDS:
        raise ValueError(f"BACKEND must be one of {', '.join(LOCAL_BACKENDS)}; got {s.backend!r}")
    if s.tts_format not in ("wav", "pcm"):
        raise ValueError(f"TTS_FORMAT must be wav or pcm; got {s.tts_format!r}")
    if s.device not in ("cpu", "cuda"):
        raise ValueError(f"DEVICE must be cpu or cuda; got {s.device!r}")
    if s.parakeet_onnx_variant not in ("fp32", "int8"):
        raise ValueError(f"PARAKEET_ONNX_VARIANT must be fp32 or int8; got {s.parakeet_onnx_variant!r}")
    if s.kokoro_onnx_variant not in ("fp32", "fp16", "int8"):
        raise ValueError(f"KOKORO_ONNX_VARIANT must be fp32, fp16 or int8; got {s.kokoro_onnx_variant!r}")
    if s.duplex.base_url and not s.duplex.base_url.startswith(("ws://", "wss://", "http://", "https://")):
        raise ValueError(f"DUPLEX_BASE_URL must start with ws://, wss://, http:// or https://; got {s.duplex.base_url!r}")
    if require_foreground and not s.foreground.base_url:
        raise ValueError("No foreground model server. Set FOREGROUND_BASE_URL (or OPENAI_BASE_URL) to any "
                         "OpenAI-compatible endpoint, e.g. http://127.0.0.1:8080/v1")
