"""Full-duplex mode without the model: the prompt and tool protocol, tool-result injection in the server, and a
session talking to a real duplex-server whose model is scripted."""
from __future__ import annotations

import asyncio
import json
import socket
from types import SimpleNamespace

import numpy as np
import pytest
import uvicorn
from conftest import FakeLLM, Harness, make_settings

from interaction_harness.config import Endpoint
from interaction_harness.duplex import (
    STREAM_SILENT,
    STREAM_VOICED,
    brief,
    parse_tool_calls,
    realtime_url,
    render_prompt,
    ungrounded,
)
from interaction_harness.duplex_server import DuplexEngine, ForcedChannel, create_app, tool_response_text
from interaction_harness.session import Session
from interaction_harness.tools import schemas

SOTC, EOTC, EOTR, PAD = 20, 21, 22, 12


def test_the_prompt_lists_duplex_tools_compactly_with_nvidias_protocol():
    p = render_prompt("Be brief.", schemas("duplex"))
    assert p.startswith("Be brief.\n\n")
    tools = json.loads(p.split("<AVAILABLE_TOOLS>")[1].split("</AVAILABLE_TOOLS>")[0])
    assert {t["name"] for t in tools} == {"get_time", "get_weather", "set_timer", "delegate_background_task"}
    assert all("description" not in v for t in tools for v in t["parameters"]["properties"].values())
    assert "<TOOLCALL>" in p and "<TOOL_RESPONSE>" in p
    assert render_prompt("Be brief.", []) == "Be brief."


def test_tool_calls_parse_with_object_or_string_arguments():
    text = ('<SPECIAL_20><TOOLCALL>[{"name": "get_weather", "arguments": {"city": "Boston"}}, '
            '{"name": "get_time", "arguments": "{}"}]</TOOLCALL><SPECIAL_21> <TOOLCALL>[{"name": "x"')
    assert parse_tool_calls(text) == [("get_weather", {"city": "Boston"}), ("get_time", {})]
    assert parse_tool_calls("<TOOLCALL>not json</TOOLCALL>") == []


def test_brief_keeps_results_short_and_prefers_a_spoken_line():
    assert brief({"iso": "2026-11-06T10:00", "spoken": "Friday, 10 AM"}) == '{"spoken": "Friday, 10 AM"}'
    long = brief({"text": "é" + "x" * 500})
    assert long.isascii() and len(long) <= 220


def test_tool_arguments_must_be_things_the_user_said():
    heard = "Stop, actually what's the weather in Boston"
    assert ungrounded({"city": "Boston"}, heard) is None
    assert ungrounded({"city": "Boston, USA"}, heard) is None
    assert ungrounded({"city": "Tokyo"}, heard) == "Tokyo"
    assert ungrounded({"task": "compare a fifteen and a thirty year mortgage on four hundred thousand"}, heard) is None
    assert ungrounded({"seconds": 15, "label": "stretch"}, "remind me in fifteen seconds to stretch") is None
    assert ungrounded({"seconds": 15, "label": "Reminder"}, "remind me in fifteen seconds to stress") is None
    assert ungrounded({"city": "London"}, "remind me in fifteen seconds to stress") == "London"


def test_realtime_url():
    ep = Endpoint("duplex", "http://box:8770/v1", "k", "")
    assert realtime_url(ep) == "ws://box:8770/v1/realtime"
    assert realtime_url(Endpoint("duplex", "wss://box/v1", "k", "")) == "wss://box/v1/realtime"


def test_forced_channel_waits_for_the_call_to_close_and_hides_noise():
    f = ForcedChannel(SOTC, EOTC)
    f.add([501, 502])
    tokens, hidden = [], []
    for predicted in (SOTC, 77, EOTC, PAD, 99, PAD, PAD):  # what the model predicts at each frame
        tokens.append(predicted)
        hidden.append(f.after_frame(tokens))
    # the call (SOTC..EOTC) is left alone; the result goes in right after it closes, one token per frame
    assert tokens == [SOTC, 77, EOTC, 501, 502, PAD, PAD]
    # the predictions it overwrote are hidden from clients; the model's own are not
    assert hidden == [False, False, False, True, True, False, False]


def test_the_model_is_held_after_a_call_until_its_result_arrives():
    f = ForcedChannel(SOTC, EOTC)
    tokens = [SOTC]
    f.after_frame(tokens)
    tokens.append(EOTC)
    f.after_frame(tokens)
    assert f.hold() and f.hold()      # nothing to read yet: skip frames rather than step past the call
    f.add([501])
    assert not f.hold()               # the result is here: the next frame reads it, right after the call
    tokens.append(PAD)
    f.after_frame(tokens)
    assert tokens[-2:] == [EOTC, 501]


def test_a_result_with_no_open_call_brings_its_call_with_it():
    engine = DuplexEngine("scripted")
    engine.parent = ScriptedParent()
    engine.open("hi")
    n = engine.add_tool_output('{"status": "fired now"}', call='[{"name": "set_timer", "arguments": {}}]')
    q = list(engine.forced.queue)
    assert q[0] == SOTC and EOTC in q and len(q) == n
    assert q[q.index(EOTC) + 1:] == FakeTokenizer().encode(tool_response_text('{"status": "fired now"}')) + [EOTR]


def test_after_a_result_the_model_must_speak_before_calling_again():
    f = ForcedChannel(SOTC, EOTC, EOTR, pad_id=PAD)
    tokens = [SOTC, EOTC]
    f.after_frame(tokens[:1])
    f.after_frame(tokens)
    f.add([501, EOTR])
    for _ in range(3):  # the result goes in
        tokens.append(PAD)
        f.after_frame(tokens)
    tokens.append(SOTC)  # the model tries to start another call right away
    hidden = f.after_frame(tokens)
    assert tokens[-1] == PAD and hidden and not f.in_call
    for _ in range(ForcedChannel.COOLDOWN_FRAMES):
        tokens.append(PAD)
        f.after_frame(tokens)
    tokens.append(SOTC)  # later, it may
    assert not f.after_frame(tokens) and tokens[-1] == SOTC and f.in_call


def test_a_call_that_never_closes_stops_blocking_results():
    f = ForcedChannel(SOTC, EOTC)
    tokens = [SOTC]
    f.after_frame(tokens)
    f.add([501])
    for _ in range(ForcedChannel.MAX_CALL_FRAMES + 1):
        tokens.append(PAD)
        f.after_frame(tokens)
    assert 501 in tokens


# --------------------------------------------------------------------------- a scripted model behind the real server
class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [1000 + i for i in range(len(text.split()))]

    def convert_tokens_to_ids(self, t):
        return {"<SPECIAL_20>": SOTC, "<SPECIAL_21>": EOTC, "<SPECIAL_22>": EOTR}[t]


def ev(kind, **kw):
    return SimpleNamespace(**{"kind": kind, "frame_index": 0, "delta": None, "text": None, "samples": None,
                              "sample_rate": None, **kw})


class ScriptedSession:
    """At frame 3 it calls get_time; from frame 12 it talks. It records the function token it reads each frame."""

    frame_samples, input_sample_rate, output_sample_rate = 1280, 16000, 22050
    config = SimpleNamespace(pad_token_id=PAD)

    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self._function_tokens: list[int] = []
        self.read: list[int] = []
        self.i = 0

    def push_audio(self, samples, sample_rate):
        self.i += 1
        if self._function_tokens:
            self.read.append(self._function_tokens[-1])
        out, token = [], PAD
        if self.i == 3:
            token = SOTC
            out.append(ev("function_delta", delta='<SPECIAL_20><TOOLCALL>[{"name": "get_time", "arguments": {}}]</TOOLCALL>'))
        elif self.i == 4:
            token = EOTC
        if self.i == 12:
            out.append(ev("assistant_text_delta", delta="It is ten."))
        loud = 12 <= self.i < 16
        out.append(ev("audio", samples=np.full(1764, 0.2 if loud else 0.0, np.float32), sample_rate=22050))
        self._function_tokens.append(token)
        return out


class ScriptedParent:
    def __init__(self):
        self.sessions: list[ScriptedSession] = []

    def create_streaming_session(self, system_prompt=None, seed=0):
        self.sessions.append(ScriptedSession())
        return self.sessions[-1]


@pytest.fixture
async def duplex_server():
    engine = DuplexEngine("scripted")
    engine.parent = ScriptedParent()
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(create_app(engine), host="127.0.0.1", port=port, log_level="warning"))
    task = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.01)
    yield engine, f"ws://127.0.0.1:{port}/v1"
    server.should_exit = True
    await task


async def pump(h: Harness, frames: int, loud: bool):
    for _ in range(frames):
        await h.frames(1, loud)
        await asyncio.sleep(0.005)  # let the link's reader run


async def test_a_session_in_duplex_mode_streams_both_ways_and_feeds_tool_results_back(duplex_server):
    engine, url = duplex_server
    h = Harness(Session, FakeLLM(), settings=make_settings(DUPLEX_BASE_URL=url), auto_play=False)
    st = await h.wait_for("duplex.status", pred=lambda e: e["state"] == "ready")
    assert st["tool_outputs"]
    await h.session.on_json({"type": "config", "mode": "duplex"})
    await h.session.on_json({"type": "mic", "on": True})
    await pump(h, 10, loud=True)    # you talk...
    await pump(h, 30, loud=False)   # ...and stop
    tc = await h.wait_for("llm.tool_call")
    assert tc["name"] == "get_time" and tc["lane"] == "duplex"
    out = await h.wait_for("duplex.tool_output")
    assert json.loads(out["output"])["spoken"]
    await pump(h, 20, loud=False)
    # the model's audio streams to the browser as one silent/voiced stream at its own rate
    ids = {cid for cid, sr, _ in h.chunks}
    assert ids == {STREAM_SILENT, STREAM_VOICED} and {sr for _, sr, _ in h.chunks} == {22_050}
    await h.session.on_json({"type": "playback", "event": "voice", "on": True})
    met = await h.wait_for("metrics", pred=lambda e: e["source"] == "duplex" and "first_audio" in e["marks"])
    assert met["marks"]["first_audio"] > 0
    # in the server, the result went into the model's function channel after the call closed, one token a frame
    session = engine.parent.sessions[-1]
    forced = FakeTokenizer().encode(tool_response_text(out["output"]))
    reads = session.read[session.read.index(EOTC) + 1:]
    start = reads.index(forced[0])
    assert reads[start:start + len(forced)] == forced
    await h.session.close()


async def test_cutting_in_measures_how_long_the_model_keeps_talking(duplex_server):
    _, url = duplex_server
    h = Harness(Session, FakeLLM(), settings=make_settings(DUPLEX_BASE_URL=url), auto_play=False)
    await h.wait_for("duplex.status", pred=lambda e: e["state"] == "ready")
    await h.session.on_json({"type": "config", "mode": "duplex"})
    await h.session.on_json({"type": "playback", "event": "voice", "on": True})   # the model is talking
    await pump(h, 8, loud=True)                                                     # you cut in
    await h.wait_for("duplex.overlap", pred=lambda e: e["who"] == "you")
    await asyncio.sleep(0.6)
    await h.session.on_json({"type": "playback", "event": "voice", "on": False})  # it stops
    y = await h.wait_for("duplex.yield")
    assert not y["kept_talking"] and 0 <= y["ms"] < 1000
    await h.session.close()


async def test_duplex_mode_needs_an_endpoint():
    h = Harness(Session, FakeLLM())
    await h.session.on_json({"type": "config", "mode": "duplex"})
    assert h.session.cfg.mode == "semantic"
    assert any(e["type"] == "log" and "DUPLEX_BASE_URL" in e["text"] for e in h.events)
    await h.session.close()


async def test_only_one_client_at_a_time(duplex_server):
    from websockets.asyncio.client import connect
    from websockets.exceptions import ConnectionClosed
    _, url = duplex_server
    msgs = []
    async with connect(url + "/realtime") as first:
        await first.send(json.dumps({"type": "session.update", "session": {"system_prompt": "hi"}}))
        async with connect(url + "/realtime") as second:
            try:
                async for m in second:
                    msgs.append(json.loads(m))
            except ConnectionClosed as e:
                assert e.rcvd.code == 1013
    assert any(m["type"] == "error" and m["error"]["code"] == "server_busy" for m in msgs)
