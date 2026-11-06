"""The state machine with every model faked: turns, tools, endpointing, barge-in, background work."""
from __future__ import annotations

import pytest
from conftest import FakeASR, FakeLLM, Harness, call, make_settings, reply

from interaction_harness.session import Config, MarkerFilter, Session


async def start(h: Harness, **cfg):
    await h.session.on_json({"type": "config", "filler": False, **cfg})


async def test_typed_turn_speaks_and_returns_to_listen():
    h = Harness(Session, FakeLLM([reply("Hello there. How are you doing today?")]))
    await start(h)
    await h.session.on_json({"type": "text", "text": "hi"})
    await h.wait_for("turn.end")
    assert h.session.state == "listen"
    assert h.session.history[-1] == {"role": "assistant", "content": "Hello there. How are you doing today?"}
    chunks = [e for e in h.events if e["type"] == "tts.chunk"]
    assert [c["text"] for c in chunks] == ["Hello there.", "How are you doing today?"]
    # the server's own sample rate travels with every chunk
    assert {sr for _, sr, _ in h.chunks} == {22_050}
    metrics = [e for e in h.events if e["type"] == "metrics" and not e["partial"]][-1]
    assert {"eot", "asr", "first_sentence", "tts_first", "first_audio", "turn_end"} <= set(metrics["marks"])
    await h.session.close()


async def test_tool_round_runs_the_tool_and_answers():
    h = Harness(Session, FakeLLM([call("calculate", expression="6*7"), reply("It's forty two.")]))
    await start(h)
    await h.session.on_json({"type": "text", "text": "what is six times seven"})
    await h.wait_for("turn.end")
    result = await h.wait_for("llm.tool_result")
    assert result["name"] == "calculate" and result["result"]["result"] == 42
    roles = [m["role"] for m in h.session.history]
    assert roles[-3:] == ["assistant", "tool", "assistant"]
    await h.session.close()


async def test_barge_in_keeps_only_what_was_heard():
    h = Harness(Session, FakeLLM([reply("First sentence here. Second one never plays. Third one neither.")]),
                auto_play=False)
    await start(h)
    await h.session.on_json({"type": "text", "text": "tell me a story"})
    first = await h.wait_for("tts.chunk")
    await h.session.on_json({"type": "playback", "event": "start", "id": first["id"]})
    await h.session.on_json({"type": "stop"})
    it = await h.wait_for("interrupt")
    assert it["spoken"] == "First sentence here."
    assert h.session.history[-1] == {"role": "assistant", "content": "First sentence here."}
    assert h.session.state == "listen"
    # the next user turn says the reply was cut off, without a marker the model could repeat
    h.llm.foreground.append(reply("Sure."))
    h.auto_play = True
    await h.session.on_json({"type": "text", "text": "go on"})
    await h.wait_for("turn.end", since=len(h.events) - 1)
    assert "cut off" in h.llm.seen[-1][-1]["content"]
    await h.session.close()


async def test_vad_mode_ends_the_turn_after_silence():
    asr = FakeASR("what time is it")
    h = Harness(Session, FakeLLM([reply("It is noon.")]), asr=asr)
    await start(h, mode="vad", silence_ms=200)
    await h.session.on_json({"type": "mic", "on": True})
    await h.frames(10, loud=True)
    assert h.session.state == "user_speaking"
    await h.frames(10, loud=False)  # 320 ms of silence
    end = await h.wait_for("user.end")
    assert end["reason"].startswith("silence")
    final = await h.wait_for("asr.final")
    assert final["text"] == "what time is it"
    await h.wait_for("turn.end")
    await h.session.close()


@pytest.mark.parametrize("partial_ms", ["640", "0"])
async def test_semantic_mode_asks_the_judge(partial_ms):
    # with partials off the judge still gets a transcript: the session transcribes on demand
    llm = FakeLLM([reply("Noon.")], complete=True)
    h = Harness(Session, llm, asr=FakeASR("what time is it"), settings=make_settings(ASR_PARTIAL_MS=partial_ms))
    await start(h, mode="semantic")
    await h.session.on_json({"type": "mic", "on": True})
    await h.frames(25, loud=True)
    await h.frames(12, loud=False)  # past sem_min_ms (300)
    end = await h.wait_for("user.end")
    assert end["reason"].startswith("semantic: COMPLETE")
    assert llm.judged and llm.judged[0] == "what time is it"
    await h.session.close()


async def test_judge_incomplete_waits_for_max_silence():
    llm = FakeLLM([reply("Okay.")], complete=False)
    h = Harness(Session, llm, asr=FakeASR("so I was wondering whether and"))
    await start(h, mode="semantic", sem_max_ms=600)
    await h.session.on_json({"type": "mic", "on": True})
    await h.frames(25, loud=True)
    await h.frames(12, loud=False)
    await h.wait_for("endpoint", pred=lambda e: e.get("verdict") == "INCOMPLETE")
    assert h.session.state == "endpointing"
    await h.frames(10, loud=False)  # now past 600 ms
    end = await h.wait_for("user.end")
    assert "600" in end["reason"]
    await h.session.close()


async def test_background_task_is_delivered_proactively():
    llm = FakeLLM(foreground=[call("delegate_background_task", task="compare two mortgages"),
                              reply("I'm on it."), reply("The fifteen year loan costs less.")],
                  background=[reply("The 15-year loan costs $275,053 less in interest.")])
    h = Harness(Session, llm)
    await start(h)
    await h.session.on_json({"type": "text", "text": "compare mortgages in the background"})
    done = await h.wait_for("background.done")
    assert "275,053" in done["result"]
    pro = await h.wait_for("proactive", timeout=5)
    assert pro["kind"] == "background"
    await h.wait_for("turn.start", pred=lambda e: e["source"] == "proactive")
    # the background event reached the foreground as a message, not as something the user said
    await h.wait_for("turn.end", pred=lambda e: e["turn"] == 2)
    assert any("Background event" in m.get("content", "") for m in llm.seen[-1] if m["role"] == "user")
    await h.session.close()


def test_config_update_coerces_strings():
    c = Config()
    c.update({"barge_in": "false", "silence_ms": "900", "speed": "1.2", "unknown": 1})
    assert c.barge_in is False and c.silence_ms == 900 and c.speed == 1.2


def test_marker_filter_catches_markers_split_across_deltas():
    f = MarkerFilter()
    out = f.push("Sure thing [interr") + f.push("upted by user] and more") + f.flush()
    assert " ".join(out.split()) == "Sure thing and more"  # the space before the marker was already sent
    f = MarkerFilter()
    assert f.push("an array [1, 2] is fine") + f.flush() == "an array [1, 2] is fine"
