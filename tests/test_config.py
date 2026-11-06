"""Four roles, their fallbacks, and where each setting comes from."""
from __future__ import annotations

import pytest

from interaction_harness.config import NO_KEY, load_settings


def settings(env=None, **kw):
    return load_settings(env={"FOREGROUND_BASE_URL": "http://fg.test/v1", **(env or {})},
                         dotenv_path=kw.pop("dotenv_path", "/nonexistent/.env"), **kw)


def test_background_falls_back_to_foreground():
    s = settings({"FOREGROUND_API_KEY": "fg-key", "FOREGROUND_MODEL": "qwen", "FOREGROUND_THINKING": "off"})
    assert (s.background.base_url, s.background.api_key, s.background.model) == ("http://fg.test/v1", "fg-key", "qwen")
    assert s.background_thinking == "off"


def test_background_on_another_server_never_gets_the_foreground_key():
    s = settings({"FOREGROUND_API_KEY": "fg-key", "FOREGROUND_MODEL": "qwen",
                  "BACKGROUND_BASE_URL": "http://bg.test:8000/v1", "BACKGROUND_MODEL": "glm"})
    assert s.background.api_key == NO_KEY
    assert s.background.model == "glm" and s.background.host == "bg.test:8000"
    assert s.background_thinking == "chat_template_kwargs"


def test_foreground_falls_back_to_openai_variables():
    s = load_settings(env={"OPENAI_BASE_URL": "https://api.example.com/v1/", "OPENAI_API_KEY": "sk-x"},
                      dotenv_path="/nonexistent/.env")
    assert s.foreground.base_url == "https://api.example.com/v1" and s.foreground.api_key == "sk-x"


def test_speech_runs_in_process_unless_given_an_endpoint():
    s = settings({"ASR_BASE_URL": "local"})
    assert s.asr.local and s.tts.local and s.asr.host == "in-process"
    s = settings({"ASR_BASE_URL": "http://fg.test/v1", "ASR_MODEL": "whisper", "FOREGROUND_API_KEY": "k",
                  "TTS_BASE_URL": "http://tts.test/v1", "TTS_MODEL": "kokoro"})
    assert not s.asr.local and s.asr.model == "whisper"
    assert s.asr.api_key == "k"          # same host as the foreground: its key may go along
    assert s.tts.api_key == NO_KEY       # a different host: it may not


def test_precedence_flags_env_dotenv_toml(tmp_path):
    toml = tmp_path / "ih.toml"
    toml.write_text('[foreground]\nmodel = "from-toml"\nthinking = "off"\n[tts]\nvoice = "bf_emma"\n'
                    '[prompts]\nforeground = "Be brief."\n')
    dotenv = tmp_path / ".env"
    dotenv.write_text("FOREGROUND_MODEL=from-dotenv\nTTS_VOICE=am_puck\n")
    s = load_settings(config_path=str(toml), dotenv_path=str(dotenv),
                      env={"FOREGROUND_BASE_URL": "http://fg.test/v1", "TTS_VOICE": "af_sky"})
    assert s.foreground.model == "from-dotenv"    # .env beats TOML
    assert s.tts_voice == "af_sky"                # env beats .env
    assert s.foreground_thinking == "off"         # TOML when nothing else says
    assert s.foreground_prompt == "Be brief."
    s = load_settings({"TTS_VOICE": "bm_lewis"}, config_path=str(toml), dotenv_path=str(dotenv),
                      env={"FOREGROUND_BASE_URL": "http://fg.test/v1", "TTS_VOICE": "af_sky"})
    assert s.tts_voice == "bm_lewis"              # flags beat everything


def test_settings_without_a_role_go_at_the_top_or_under_server(tmp_path):
    toml = tmp_path / "ih.toml"
    toml.write_text('host = "0.0.0.0"\n[server]\nport = 9001\ntools = ["a.py", "b.py"]\n')
    s = settings(config_path=str(toml))
    assert (s.host, s.port, s.tools_files) == ("0.0.0.0", 9001, ["a.py", "b.py"])


@pytest.mark.parametrize("env, message", [
    ({"FOREGROUND_THINKING": "maybe"}, "FOREGROUND_THINKING"),
    ({"BACKEND": "cuda"}, "BACKEND"),
    ({"TTS_FORMAT": "mp3"}, "TTS_FORMAT"),
    ({"KOKORO_ONNX_VARIANT": "int4"}, "KOKORO_ONNX_VARIANT"),
])
def test_bad_values_say_which_setting(env, message):
    with pytest.raises(ValueError, match=message):
        settings(env)


def test_foreground_is_required_except_for_warmup():
    with pytest.raises(ValueError, match="FOREGROUND_BASE_URL"):
        load_settings(env={}, dotenv_path="/nonexistent/.env")
    s = load_settings(env={}, dotenv_path="/nonexistent/.env", require_foreground=False)
    assert s.asr.local


def test_missing_config_file_is_an_error():
    with pytest.raises(FileNotFoundError):
        settings(config_path="/nonexistent/ih.toml")


def test_duplex_is_off_unless_given_an_endpoint_and_never_gets_another_roles_key():
    s = settings({"FOREGROUND_API_KEY": "fg-key"})
    assert s.duplex.base_url is None and "duplex" not in s.roles()
    s = settings({"FOREGROUND_API_KEY": "fg-key", "DUPLEX_BASE_URL": "ws://box:8770/v1/"})
    assert (s.duplex.base_url, s.duplex.api_key) == ("ws://box:8770/v1", NO_KEY)
    assert "duplex" in s.roles()
    with pytest.raises(ValueError, match="DUPLEX_BASE_URL"):
        settings({"DUPLEX_BASE_URL": "box:8770"})
