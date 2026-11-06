"""Speech pieces that need no model: text chunking, the VAD wrapper, and the endpoint clients."""
from __future__ import annotations

import io
from pathlib import Path

import httpx
import numpy as np
import pytest
import soundfile as sf

from interaction_harness.config import Endpoint
from interaction_harness.speech.base import resample
from interaction_harness.speech.endpoint import EndpointASR, EndpointTTS
from interaction_harness.speech.silero import SileroVAD
from interaction_harness.speech.text import SentenceChunker, strip_markdown

DATA = Path(__file__).parent / "data"


def test_chunker_cuts_the_first_chunk_early_at_a_clause():
    c = SentenceChunker()
    out = []
    for tok in "Well, after thinking about it for quite a while, I'd say the answer is yes. Then more.".split(" "):
        out += c.push(tok + " ")
    out += c.flush()
    assert out[0] == "Well, after thinking about it for quite a while,"
    assert out[1:] == ["I'd say the answer is yes.", "Then more."]


def test_chunker_merges_very_short_sentences():
    c = SentenceChunker()
    assert c.push("Yes. ") == []
    assert c.push("That works for me. ") == ["Yes. That works for me."]


def test_chunker_does_not_split_a_decimal_across_tokens():
    c = SentenceChunker()
    out = []
    for tok in ["The 15-year loan at 5", ".", "5% costs less.", " Then more."]:
        out += c.push(tok)
    out += c.flush()
    assert out == ["The 15-year loan at 5.5% costs less.", "Then more."]


def test_strip_markdown():
    assert strip_markdown("**Bold** and `code` and [a link](https://x.y) https://z.w") == "Bold and code and a link"


def test_silero_matches_the_reference_wrapper():
    # probabilities from silero-vad's own (torch) OnnxWrapper on the same clip
    clip, ref = np.load(DATA / "vad_clip.npy"), np.load(DATA / "vad_ref.npy")
    vad = SileroVAD()
    got = np.array([vad.prob(clip[i:i + 512]) for i in range(0, len(clip) - 512 + 1, 512)])
    assert np.abs(got - ref).max() < 1e-4
    vad.reset()
    assert max(vad.prob(np.zeros(512, dtype=np.int16)) for _ in range(5)) < 0.1
    with pytest.raises(ValueError):
        vad.prob(np.zeros(100, dtype=np.int16))


def test_resample():
    x = np.arange(24_000, dtype=np.int16)
    assert len(resample(x, 24_000, 16_000)) == 16_000 and resample(x, 16_000, 16_000) is x


def ep(**kw):
    return Endpoint(kw.pop("role", "asr"), "http://speech.test/v1", "secret", kw.pop("model", "m"))


async def test_endpoint_asr_posts_a_wav_and_reads_json_or_text():
    seen = {}

    def handler(request: httpx.Request):
        seen["auth"] = request.headers["authorization"]
        seen["path"] = request.url.path
        body = request.read()
        seen["has_wav"] = b"RIFF" in body and b'name="model"' in body and b"json" in body
        if seen.get("plain"):
            return httpx.Response(200, text="plain words")
        return httpx.Response(200, json={"text": " hello world "})

    asr = EndpointASR(ep(), transport=httpx.MockTransport(handler))
    text, _ = await asr.transcribe(np.zeros(16_000, dtype=np.float32))
    assert text == "hello world"
    assert seen == {"auth": "Bearer secret", "path": "/v1/audio/transcriptions", "has_wav": True}
    seen["plain"] = True
    assert (await asr.transcribe(np.zeros(16_000, dtype=np.float32)))[0] == "plain words"
    assert (await asr.transcribe(np.zeros(100, dtype=np.float32)))[0] == ""  # too short to send


async def test_endpoint_asr_partials_retranscribe_the_turn_so_far():
    sizes = []

    def handler(request):
        sizes.append(len(request.read()))
        return httpx.Response(200, json={"text": "partial"})

    stream = await EndpointASR(ep(), transport=httpx.MockTransport(handler)).stream()
    for _ in range(3):
        await stream.feed(np.zeros(8_000, dtype=np.float32))
    assert stream.text == "partial" and sizes[0] < sizes[1] < sizes[2]


async def test_endpoint_errors_name_the_status():
    asr = EndpointASR(ep(), transport=httpx.MockTransport(lambda r: httpx.Response(503, text="overloaded")))
    with pytest.raises(RuntimeError, match="HTTP 503"):
        await asr.transcribe(np.zeros(16_000, dtype=np.float32))


async def test_endpoint_tts_wav_keeps_the_servers_rate_and_voices_are_discovered():
    buf = io.BytesIO()
    sf.write(buf, np.zeros(22_050, dtype=np.int16), 22_050, format="WAV", subtype="PCM_16")

    def handler(request: httpx.Request):
        if request.url.path.endswith("/audio/voices"):
            assert request.url.params["model"] == "kokoro"
            return httpx.Response(200, json={"data": [{"id": "af_alloy"}, {"id": "af_heart"}, {"id": "zz"}]})
        assert request.read() and b'"response_format":"wav"' in request.content.replace(b" ", b"")
        return httpx.Response(200, content=buf.getvalue())

    tts = EndpointTTS(ep(role="tts", model="kokoro"), transport=httpx.MockTransport(handler))
    assert await tts.discover_voices() == ["af_heart", "af_alloy", "zz"]
    pcm, sr, _ = await tts.synth("hello", "af_heart")
    assert sr == 22_050 and len(pcm) == 22_050 and pcm.dtype == np.int16


async def test_endpoint_tts_pcm_uses_the_configured_rate():
    tts = EndpointTTS(ep(role="tts"), fmt="pcm", pcm_rate=24_000,
                      transport=httpx.MockTransport(lambda r: httpx.Response(200, content=b"\x01\x00" * 480)))
    pcm, sr, _ = await tts.synth("hi", "alloy")
    assert sr == 24_000 and len(pcm) == 480


def test_endpoint_speech_needs_a_model():
    with pytest.raises(ValueError, match="ASR_MODEL"):
        EndpointASR(ep(model=""))
    with pytest.raises(ValueError, match="TTS_MODEL"):
        EndpointTTS(ep(role="tts", model=""))
