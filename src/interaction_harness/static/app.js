/* interaction-harness dashboard: vanilla JS, no build step. */
(() => {
  const $ = (s) => document.querySelector(s);
  const fmtMs = (v) => (v == null ? '—' : v >= 1000 ? (v / 1000).toFixed(2) + ' s' : Math.round(v) + ' ms');
  const esc = (s) => String(s ?? '').replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));

  // ---------------------------------------------------------------- socket
  let ws, wsOpen = false;
  const send = (o) => { if (wsOpen) ws.send(JSON.stringify(o)); };
  function connect() {
    ws = new WebSocket((location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/ws');
    ws.binaryType = 'arraybuffer';
    ws.onopen = () => { wsOpen = true; $('#pill-ws').textContent = 'socket · open'; $('#pill-ws').className = 'pill ws on'; pushConfig(); };
    ws.onclose = () => { wsOpen = false; $('#pill-ws').textContent = 'socket · closed (retrying)'; $('#pill-ws').className = 'pill ws off'; setTimeout(connect, 1500); };
    ws.onmessage = (ev) => { if (ev.data instanceof ArrayBuffer) onAudio(ev.data); else onEvent(JSON.parse(ev.data)); };
  }

  // ---------------------------------------------------------------- config
  const cfg = { mode: 'semantic', barge_in: true, filler: true, foreground_thinking: false, voice: '', speed: 1.05, silence_ms: 700 };
  function pushConfig() { const c = { type: 'config', ...cfg }; if (!c.voice) delete c.voice; send(c); applyModeUI(); }
  function applyModeUI() {
    const duplex = cfg.mode === 'duplex';
    document.querySelectorAll('#mode button').forEach((b) => b.classList.toggle('on', b.dataset.mode === cfg.mode));
    $('#btn-ptt').hidden = cfg.mode !== 'ptt';
    $('#lbl-silence').style.display = cfg.mode === 'vad' ? '' : 'none';
    ['#cfg-barge', '#cfg-filler', '#cfg-think', '#cfg-voice', '#cfg-speed'].forEach((id) => { $(id).disabled = duplex; });
    $('.panel.brain h2 small').textContent = duplex ? 'full-duplex model · hears and talks at once · tools · delegation'
      : `foreground · thinking ${cfg.foreground_thinking ? 'on' : 'off'} · tools · delegation`;
    $('.panel.ear h2 small').textContent = duplex ? 'Silero VAD (measuring only) → the model decides' : 'Silero VAD → ASR → end-of-turn';
    $('.panel.mouth h2 small').textContent = duplex ? 'the model\'s own voice, streamed every 80 ms' : 'sentence-chunked TTS → browser playback';
  }
  document.querySelectorAll('#mode button').forEach((b) => b.onclick = () => { cfg.mode = b.dataset.mode; pushConfig(); });
  $('#cfg-barge').onchange = (e) => { cfg.barge_in = e.target.checked; pushConfig(); };
  $('#cfg-filler').onchange = (e) => { cfg.filler = e.target.checked; pushConfig(); };
  $('#cfg-think').onchange = (e) => { cfg.foreground_thinking = e.target.checked; pushConfig(); };
  $('#cfg-voice').onchange = (e) => { cfg.voice = e.target.value; pushConfig(); };
  $('#cfg-speed').onchange = (e) => { cfg.speed = +e.target.value; pushConfig(); };
  $('#cfg-silence').onchange = (e) => { cfg.silence_ms = +e.target.value; pushConfig(); };
  $('#btn-stop').onclick = () => { send({ type: 'stop' }); player.stop(); };
  $('#btn-reset').onclick = () => { send({ type: 'reset' }); $('#convo').innerHTML = ''; $('#finals').innerHTML = ''; $('#chunks').innerHTML = ''; $('#bg-list').innerHTML = '<div class="empty">Cleared.</div>'; };
  $('#typed').onsubmit = (e) => { e.preventDefault(); const t = $('#typed-in').value.trim(); if (t) { send({ type: 'text', text: t }); $('#typed-in').value = ''; } };

  // ---------------------------------------------------------------- mic in
  let micCtx, micNode, micStream, micOn = false, onMicStream = null;
  const wave = { rms: new Float32Array(240), vad: new Float32Array(240), i: 0 };
  async function startMic() {
    micStream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true } });
    micCtx = new AudioContext();
    await micCtx.audioWorklet.addModule('/static/mic-worklet.js');
    const src = micCtx.createMediaStreamSource(micStream);
    micNode = new AudioWorkletNode(micCtx, 'mic-processor');
    micNode.port.onmessage = (e) => {
      if (wsOpen) ws.send(e.data.pcm);
      wave.rms[wave.i] = e.data.rms; wave.i = (wave.i + 1) % wave.rms.length;
    };
    src.connect(micNode); // no connection to destination: we never play the mic
    if (onMicStream) onMicStream();
    micOn = true; $('#btn-mic').textContent = '⏹ Stop mic'; $('#btn-mic').classList.add('on');
    send({ type: 'mic', on: true });
    player.ensure();
  }
  function stopMic() {
    micNode && micNode.disconnect(); micStream && micStream.getTracks().forEach((t) => t.stop()); micCtx && micCtx.close();
    micOn = false; $('#btn-mic').textContent = '🎙 Start mic'; $('#btn-mic').classList.remove('on');
    send({ type: 'mic', on: false });
  }
  $('#btn-mic').onclick = () => (micOn ? stopMic() : startMic().catch((e) => log({ type: 'log', level: 'error', text: 'mic: ' + e.message })));

  // push-to-talk (button or space bar)
  let pttDown = false;
  const ptt = (down) => { if (cfg.mode !== 'ptt' || !micOn || down === pttDown) return; pttDown = down; $('#btn-ptt').classList.toggle('down', down); send({ type: 'ptt', down }); };
  const pb = $('#btn-ptt');
  pb.onpointerdown = (e) => { e.preventDefault(); pb.setPointerCapture(e.pointerId); ptt(true); };
  pb.onpointerup = pb.onpointercancel = () => ptt(false);
  window.addEventListener('keydown', (e) => { if (e.code === 'Space' && document.activeElement.tagName !== 'INPUT') { e.preventDefault(); ptt(true); } });
  window.addEventListener('keyup', (e) => { if (e.code === 'Space') ptt(false); });

  // ---------------------------------------------------------------- audio out
  // TTS chunks are acknowledged one by one (start/end), so the server knows what you heard. The full-duplex
  // model's audio is one continuous stream of 80 ms frames: played gaplessly behind a small jitter buffer, and
  // only its silent -> voiced -> silent changes are reported, at the moment you hear them.
  const STREAM_SILENT = 0xFFFFFFFE, STREAM_VOICED = 0xFFFFFFFF, JITTER_S = 0.12;
  const player = {
    ctx: null, analyser: null, next: 0, sources: new Map(), pending: 0, streamNext: 0, streamSources: new Set(), voiced: false,
    ensure() {
      if (this.ctx) { if (this.ctx.state === 'suspended') this.ctx.resume(); return; }
      this.ctx = new AudioContext();
      this.analyser = this.ctx.createAnalyser(); this.analyser.fftSize = 512; this.analyser.connect(this.ctx.destination);
      const data = new Float32Array(this.analyser.fftSize);
      const tick = () => { this.analyser.getFloatTimeDomainData(data); let s = 0; for (const v of data) s += v * v; const r = Math.sqrt(s / data.length); $('#out-bar').style.width = Math.min(100, r * 300) + '%'; requestAnimationFrame(tick); };
      tick();
    },
    play(id, sr, pcm) {
      this.ensure();
      const buf = this.ctx.createBuffer(1, pcm.length, sr); const ch = buf.getChannelData(0);
      for (let i = 0; i < pcm.length; i++) ch[i] = pcm[i] / 32768;
      const src = this.ctx.createBufferSource(); src.buffer = buf; src.connect(this.analyser);
      const now = this.ctx.currentTime; const at = Math.max(now, this.next); this.next = at + buf.duration;
      src.start(at); this.sources.set(id, src); this.pending++;
      setTimeout(() => { if (this.sources.has(id)) { send({ type: 'playback', event: 'start', id }); chunkState(id, 'playing'); } }, Math.max(0, (at - now) * 1000));
      src.onended = () => { if (!this.sources.has(id)) return; this.sources.delete(id); this.pending--; chunkState(id, 'done'); send({ type: 'playback', event: 'end', id }); if (this.pending === 0) send({ type: 'playback', event: 'drained' }); };
    },
    playStream(sr, pcm, voiced) {
      this.ensure();
      const buf = this.ctx.createBuffer(1, pcm.length, sr); const ch = buf.getChannelData(0);
      for (let i = 0; i < pcm.length; i++) ch[i] = pcm[i] / 32768;
      const src = this.ctx.createBufferSource(); src.buffer = buf; src.connect(this.analyser);
      const now = this.ctx.currentTime;
      if (this.streamNext < now) this.streamNext = now + JITTER_S;   // underrun: rebuild the buffer
      const at = this.streamNext; this.streamNext += buf.duration;
      src.start(at); this.streamSources.add(src); src.onended = () => this.streamSources.delete(src);
      if (voiced !== this.voiced) { this.voiced = voiced; setTimeout(() => send({ type: 'playback', event: 'voice', on: voiced }), Math.max(0, (at - now) * 1000)); }
    },
    stop() {
      for (const [id, s] of this.sources) { try { s.stop(); } catch (_) {} chunkState(id, 'dropped'); } this.sources.clear(); this.pending = 0; this.next = 0;
      for (const s of this.streamSources) { try { s.stop(); } catch (_) {} } this.streamSources.clear(); this.streamNext = 0;
      if (this.voiced) { this.voiced = false; send({ type: 'playback', event: 'voice', on: false }); }
    }
  };
  function onAudio(ab) {
    const dv = new DataView(ab); const id = dv.getUint32(0, true), sr = dv.getUint32(4, true);
    if (id === STREAM_SILENT || id === STREAM_VOICED) { player.playStream(sr, new Int16Array(ab, 8), id === STREAM_VOICED); return; }
    player.play(id, sr, new Int16Array(ab, 8));
    chunkState(id, 'ready');
  }

  // ---------------------------------------------------------------- recording (open the page with ?record)
  // This tab's video, with your mic and the agent's voice mixed into one track, saved as a .webm.
  if (new URLSearchParams(location.search).has('record')) {
    const btn = document.createElement('button'); btn.className = 'btn'; btn.textContent = '⏺ Record';
    $('.toolbar').insertBefore(btn, $('#btn-stop'));
    let rec = null, mix = null, micTap = null, agentGain = null, parts = [];
    const gain = (v) => { const g = player.ctx.createGain(); g.gain.value = v; g.connect(mix); return g; };  // headroom: two voices can overlap
    onMicStream = () => { if (!mix) return; if (micTap) micTap.disconnect(); micTap = player.ctx.createMediaStreamSource(micStream); micTap.connect(gain(0.7)); };
    btn.onclick = async () => {
      if (rec) { rec.stop(); return; }
      let screen;
      try { screen = await navigator.mediaDevices.getDisplayMedia({ video: { frameRate: 30 }, audio: false, preferCurrentTab: true, selfBrowserSurface: 'include' }); }
      catch (e) { log({ type: 'log', level: 'error', text: 'record: ' + e.message }); return; }
      player.ensure(); mix = player.ctx.createMediaStreamDestination(); agentGain = gain(0.7); player.analyser.connect(agentGain);
      if (micStream) onMicStream();
      const type = ['video/webm;codecs=vp9,opus', 'video/webm;codecs=vp8,opus', 'video/webm'].find((t) => MediaRecorder.isTypeSupported(t));
      parts = []; rec = new MediaRecorder(new MediaStream([...screen.getVideoTracks(), ...mix.stream.getAudioTracks()]), { mimeType: type });
      rec.ondataavailable = (e) => { if (e.data.size) parts.push(e.data); };
      rec.onstop = () => {
        screen.getTracks().forEach((t) => t.stop()); player.analyser.disconnect(agentGain); if (micTap) micTap.disconnect(); micTap = mix = rec = agentGain = null;
        const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob(parts, { type: 'video/webm' }));
        a.download = `interaction-harness-${new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-')}.webm`; a.click();
        btn.textContent = '⏺ Record'; btn.classList.remove('warn');
      };
      screen.getVideoTracks()[0].onended = () => { if (rec) rec.stop(); };
      rec.start(1000); btn.textContent = '⏹ Stop recording'; btn.classList.add('warn');
    };
  }

  // ---------------------------------------------------------------- waveform
  const cv = $('#wave'); const cx = cv.getContext('2d');
  function drawWave() {
    const W = cv.width = cv.clientWidth * devicePixelRatio, H = cv.height = 90 * devicePixelRatio;
    cx.clearRect(0, 0, W, H); const n = wave.rms.length, bw = W / n;
    for (let k = 0; k < n; k++) {
      const j = (wave.i + k) % n; const h = Math.min(H, wave.rms[j] * H * 4);
      const p = wave.vad[j]; cx.fillStyle = p > 0.5 ? '#3987e5' : '#4a4a52';
      cx.fillRect(k * bw, (H - h) / 2, Math.max(1, bw - 1), Math.max(1, h));
    }
    requestAnimationFrame(drawWave);
  }
  drawWave();

  // ---------------------------------------------------------------- UI helpers
  const convo = $('#convo');
  let curAssistant = null, curThink = null, curTurnSource = 'voice', userStartedAt = 0;
  const capChildren = (el, n) => { while (el.children.length > n) el.firstChild.remove(); };
  function bubble(role, text, cls = '') { const d = document.createElement('div'); d.className = `msg ${role} ${cls}`; d.textContent = text; convo.appendChild(d); capChildren(convo, 300); convo.scrollTop = convo.scrollHeight; return d; }
  function assistantBubble(source) {
    curAssistant = document.createElement('div'); curAssistant.className = 'msg assistant' + (source === 'proactive' ? ' proactive' : ''); curAssistant.dataset.t = Date.now();
    const tag = document.createElement('span'); tag.className = 'tag'; tag.textContent = source === 'proactive' ? 'proactive · agent-initiated turn' : 'assistant';
    curAssistant.appendChild(tag); curAssistant.appendChild(document.createTextNode('')); convo.appendChild(curAssistant); curThink = null;
  }
  function thinkBlock(lane) {
    const d = document.createElement('details'); d.className = 'think'; d.open = true;
    d.innerHTML = `<summary>${lane} · reasoning</summary><pre></pre>`; convo.appendChild(d); return d.querySelector('pre');
  }
  function chunkState(id, st) {
    const li = document.getElementById('chunk-' + id); if (!li) return;
    li.classList.remove('queued', 'synth', 'ready', 'playing', 'done', 'dropped'); li.classList.add(st);
    if (st === 'playing') li.scrollIntoView({ block: 'nearest' });
  }
  const logEl = $('#log');
  function log(ev) {
    const d = document.createElement('div'); d.className = ev.level || '';
    const { type, t, ...rest } = ev; delete rest.delta;
    const body = ev.text ?? Object.entries(rest).map(([k, v]) => `${k}=${typeof v === 'object' ? JSON.stringify(v) : v}`).join(' ');
    d.innerHTML = `<span class="t">${new Date(t || Date.now()).toLocaleTimeString([], { hour12: false })}.${String((t || 0) % 1000).padStart(3, '0')}</span><span class="ty">${esc(type)}</span>${esc(body).slice(0, 400)}`;
    logEl.appendChild(d); while (logEl.children.length > 400) logEl.firstChild.remove(); logEl.scrollTop = logEl.scrollHeight;
  }

  // ---------------------------------------------------------------- waterfall (latency)
  const STAGES = [
    { k: 'endpoint', from: null, to: 'eot', c: '#3987e5', l: 'endpointing (silence + judge)' },
    { k: 'asr', from: 'eot', to: 'asr', c: '#d95926', l: 'ASR final' },
    { k: 'llm', from: 'asr', to: 'first_sentence', c: '#199e70', l: 'foreground → first sentence' },
    { k: 'tts', from: 'first_sentence', to: 'tts_first', c: '#c98500', l: 'TTS first chunk' },
    { k: 'play', from: 'tts_first', to: 'first_audio', c: '#d55181', l: 'transport → playback' },
  ];
  const DUPLEX_STAGES = [{ k: 'duplex', from: null, to: 'first_audio', c: '#8b6fd6', l: 'full-duplex model → first audio' }];
  $('#legend').innerHTML = [...STAGES, ...DUPLEX_STAGES].map((s) => `<span><i style="background:${s.c}"></i>${s.l}</span>`).join('');
  const turns = new Map();
  function renderWaterfall() {
    const rows = [...turns.values()].slice(-8);
    if (!rows.length) return;
    const W = $('#waterfall').clientWidth || 400, rowH = 22, left = 44, right = 70, top = 14;
    const maxMs = Math.max(1000, ...rows.map((m) => m.marks.first_audio ?? m.marks.turn_end ?? m.marks.interrupted ?? 0));
    const x = (ms) => left + (ms / maxMs) * (W - left - right);
    const H = top + rows.length * rowH + 16;
    let s = `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}"><g class="grid">`;
    const step = maxMs > 4000 ? 1000 : maxMs > 1500 ? 500 : 250;
    for (let ms = 0; ms <= maxMs; ms += step) s += `<line x1="${x(ms)}" x2="${x(ms)}" y1="${top - 4}" y2="${H - 12}"/><text x="${x(ms)}" y="${H - 2}" text-anchor="middle">${ms >= 1000 ? (ms / 1000) + 's' : ms}</text>`;
    s += '</g>';
    rows.forEach((m, r) => {
      const y = top + r * rowH;
      s += `<text x="${left - 6}" y="${y + 13}" text-anchor="end">#${m.turn}${m.source === 'typed' ? '⌨' : m.source === 'proactive' ? '★' : m.source === 'duplex' ? '⇄' : ''}</text>`;
      let prev = 0;
      for (const st of m.source === 'duplex' ? DUPLEX_STAGES : STAGES) {
        const end = m.marks[st.to]; if (end == null) continue;
        const start = st.from ? (m.marks[st.from] ?? prev) : 0;
        if (end > start) s += `<rect class="seg" x="${x(start)}" y="${y + 2}" width="${Math.max(2, x(end) - x(start) - 1)}" height="${rowH - 6}" fill="${st.c}" data-tip="${esc(st.l)}: ${Math.round(end - start)} ms (${Math.round(start)}→${Math.round(end)})"></rect>`;
        prev = end;
      }
      const ttfa = m.marks.first_audio;
      const lbl = m.marks.overlap ? 'overlap' : ttfa != null ? `TTFA ${fmtMs(ttfa)}` : m.interrupted ? 'cut off' : m.marks.turn_end ? 'no audio' : '…';
      s += `<text x="${W - right + 6}" y="${y + 13}">${lbl}${m.interrupted ? ' ✂' : ''}</text>`;
    });
    $('#waterfall').innerHTML = s + '</svg>';
  }
  const tip = $('#tip');
  $('#waterfall').addEventListener('mousemove', (e) => { const r = e.target.closest('rect.seg'); if (!r) { tip.hidden = true; return; } tip.hidden = false; tip.textContent = r.dataset.tip; tip.style.left = e.clientX + 12 + 'px'; tip.style.top = e.clientY + 12 + 'px'; });
  $('#waterfall').addEventListener('mouseleave', () => (tip.hidden = true));

  // ---------------------------------------------------------------- background model
  const bgList = $('#bg-list');
  function taskCard(id, cls, title, why, fresh = false) {
    let c = document.getElementById('task-' + id);
    if (c && fresh) { c.remove(); c = null; }
    if (c) return c;
    if (bgList.querySelector('.empty')) bgList.innerHTML = '';
    c = document.createElement('div'); c.id = 'task-' + id; c.className = `task ${cls} running`;
    c.innerHTML = `<div class="hd"><span class="id">${title}</span><span class="status">running</span></div><div class="why">${esc(why)}</div><div class="tools"></div><details open><summary style="color:var(--muted);font-size:11px;cursor:pointer">reasoning</summary><pre></pre></details><div class="result"></div>`;
    bgList.prepend(c); return c;
  }
  const setStatus = (c, st, cls) => { c.querySelector('.status').textContent = st; c.classList.remove('running', 'done', 'delivered'); c.classList.add(cls); };

  // ---------------------------------------------------------------- events
  let epTimer = null, epSilenceStart = 0;
  const handlers = {
    hello(ev) {
      if (handlers._seenHello) { $('#convo').innerHTML = ''; $('#chunks').innerHTML = ''; $('#finals').innerHTML = ''; bgList.innerHTML = '<div class="empty">Reconnected: new session.</div>'; turns.clear(); $('#waterfall').innerHTML = '<div class="empty">No turns yet.</div>'; }
      handlers._seenHello = true;
      $('#pill-duplex').hidden = $('#mode-duplex').hidden = !ev.roles.duplex;
      for (const [role, r] of Object.entries(ev.roles)) {
        const pill = $('#pill-' + role); if (!pill) continue;
        pill.textContent = `${role === 'asr' || role === 'tts' ? role.toUpperCase() : role} · ${r.model.split('/').pop()}`;
        pill.title = `${pill.title.split(' — ')[0]} — ${r.model} @ ${r.where}`;
      }
      if (!cfg.voice || !ev.voices.includes(cfg.voice)) { cfg.voice = ev.voice || ev.voices[0]; send({ type: 'config', voice: cfg.voice }); }
      const sel = $('#cfg-voice'); sel.innerHTML = ev.voices.map((v) => `<option ${v === cfg.voice ? 'selected' : ''}>${esc(v)}</option>`).join('');
      const say = new URLSearchParams(location.search).get('say');   // ?say=... auto-sends a typed turn (handy for demos/tests)
      if (say) setTimeout(() => send({ type: 'text', text: say }), 300);
    },
    config(ev) { $('#vad-thresh').style.left = ev.onset_thresh * 100 + '%'; },
    state(ev) {
      document.querySelectorAll('#states span').forEach((s) => s.classList.toggle('on', s.dataset.s === ev.state));
      if (ev.state === 'endpointing') { epSilenceStart = performance.now(); clearInterval(epTimer); epTimer = setInterval(() => { const ms = performance.now() - epSilenceStart + (cfg.mode === 'vad' ? cfg.silence_ms : 300); const max = cfg.mode === 'vad' ? cfg.silence_ms : 2200; $('#ep-fill').style.width = Math.min(100, ms / max * 100) + '%'; $('#ep-ms').textContent = Math.round(ms) + ' ms'; }, 40); }
      else { clearInterval(epTimer); if (ev.state !== 'user_speaking') { $('#ep-fill').style.width = '0'; $('#ep-ms').textContent = '0 ms'; } }
      if (ev.state === 'user_speaking') { $('#ep-verdict').textContent = '—'; $('#ep-fill').style.width = '0'; }
    },
    vad(ev) { $('#vad-bar').style.width = ev.p * 100 + '%'; $('#vad-lbl').textContent = 'VAD ' + ev.p.toFixed(2); wave.vad[(wave.i + wave.vad.length - 1) % wave.vad.length] = ev.p; },
    'user.start'() { $('#partial').textContent = '…'; userStartedAt = Date.now(); },
    'asr.partial'(ev) { $('#partial').textContent = (ev.text || '…') + `  ·  ${ev.audio_s}s · ${Math.round(ev.ms)} ms`; },
    'asr.final'(ev) {
      $('#partial').textContent = '';
      const li = document.createElement('li'); li.innerHTML = `${esc(ev.text)}<span class="ms">${ev.model === 'typed' ? 'typed' : ev.model === 'duplex' ? 'as the duplex model heard it' : `${ev.audio_s}s audio → ${Math.round(ev.ms)} ms`}</span>`;
      $('#finals').prepend(li); capChildren($('#finals'), 60);
      if (ev.text.startsWith('(Background event')) return;
      const b = bubble('user', ev.text);
      // full-duplex: the model may have answered before you finished; put what you said where you said it
      const answer = ev.model === 'duplex' && [...convo.querySelectorAll('.msg.assistant')].find((d) => +d.dataset.t >= userStartedAt);
      if (answer) convo.insertBefore(b, answer);
    },
    endpoint(ev) {
      if (ev.phase === 'judge') $('#ep-verdict').textContent = `${ev.verdict}  (${Math.round(ev.ms || 0)} ms, after ${ev.silence_ms ?? '?'} ms silence)  “${ev.text || ''}”`;
      else if (ev.phase === 'silence') $('#ep-verdict').textContent = ev.verdict;
      else if (ev.phase === 'resumed') $('#ep-verdict').textContent = 'user resumed — not done yet';
    },
    'turn.start'(ev) { curTurnSource = ev.source; assistantBubble(ev.source); if (ev.source !== 'duplex') { turns.set(ev.turn, { turn: ev.turn, source: ev.source, marks: {} }); renderWaterfall(); } },
    'llm.first_token'(ev) { $('#llm-stat').textContent = `TTFT ${Math.round(ev.ms)} ms`; },
    'llm.reasoning'(ev) { if (!curThink) curThink = thinkBlock('foreground'); curThink.textContent += ev.delta; curThink.scrollTop = curThink.scrollHeight; },
    'llm.content'(ev) { if (!curAssistant) assistantBubble(curTurnSource); curAssistant.lastChild.textContent += ev.delta; convo.scrollTop = convo.scrollHeight; },
    'llm.tool_call'(ev) {
      const d = document.createElement('div'); d.className = 'tool' + (['set_timer', 'delegate_background_task'].includes(ev.name) ? ' async' : ''); d.id = 'tool-' + ev.id;
      d.innerHTML = `<span class="name">${esc(ev.name)}</span>(<span class="args">${esc(ev.arguments)}</span>) <span class="ms">…</span><div class="res"></div>`;
      convo.appendChild(d); curAssistant = null; convo.scrollTop = convo.scrollHeight;
    },
    'llm.tool_result'(ev) { const d = document.getElementById('tool-' + ev.id); if (!d) return; d.querySelector('.ms').textContent = Math.round(ev.ms) + ' ms'; d.querySelector('.res').textContent = typeof ev.result === 'string' ? ev.result : JSON.stringify(ev.result, null, 1).slice(0, 600); },
    'llm.done'(ev) { $('#llm-stat').textContent = `${$('#llm-stat').textContent.split('·')[0].trim()} · ${ev.tokens} tok · ${ev.tps} tok/s`; },
    'tts.queued'(ev) { const li = document.createElement('li'); li.id = 'chunk-pending-' + ev.text.length + '-' + Date.now(); li.className = 'queued'; li.dataset.text = ev.text; li.innerHTML = `<span class="st"></span><span class="txt">${esc(ev.text)}</span><span class="m">queued</span>`; $('#chunks').appendChild(li); capChildren($('#chunks'), 80); li.scrollIntoView({ block: 'nearest' }); },
    'tts.chunk'(ev) {
      let li = [...$('#chunks').querySelectorAll('li.queued')].find((l) => l.dataset.text === ev.text);
      if (!li) { li = document.createElement('li'); li.innerHTML = `<span class="st"></span><span class="txt">${esc(ev.text)}</span><span class="m"></span>`; $('#chunks').appendChild(li); }
      li.id = 'chunk-' + ev.id; li.className = 'synth' + (ev.kind === 'filler' ? ' filler' : '');
      li.querySelector('.m').textContent = ev.kind === 'filler' ? 'filler (cached)' : `${Math.round(ev.synth_ms)} ms → ${ev.audio_s}s · RTF ${ev.rtf}`;
    },
    'playback.stop'() { player.stop(); },
    interrupt(ev) {
      if (curAssistant) { const c = document.createElement('span'); c.className = 'cut'; c.textContent = ` ✂ interrupted (${ev.source})` + (ev.dropped ? ` — dropped: “${ev.dropped.slice(0, 80)}${ev.dropped.length > 80 ? '…' : ''}”` : ''); curAssistant.appendChild(c); curAssistant.classList.add('interrupted'); }
      curAssistant = null; curThink = null; $('#chunks').querySelectorAll('li.queued,li.synth,li.ready').forEach((l) => l.className = 'dropped');
    },
    metrics(ev) { const m = turns.get(ev.turn) || { turn: ev.turn, source: ev.source, marks: {} }; Object.assign(m, ev); turns.set(ev.turn, m); renderWaterfall(); if (ev.marks.first_audio != null && ev.source !== 'proactive') $('#ttfa').textContent = `last TTFA ${fmtMs(ev.marks.first_audio)}`; },
    'turn.end'() { curAssistant = null; curThink = null; },
    'background.queued'(ev) { const c = taskCard(ev.id, 'background', `background task #${ev.id}`, ev.task, true); c.querySelector('.status').textContent = 'queued behind another task'; },
    'background.start'(ev) { const c = taskCard(ev.id, 'background', `background task #${ev.id}`, ev.task, true); c.scrollIntoView({ block: 'start' }); capChildren(bgList, 40); },
    'background.reasoning'(ev) { const c = taskCard(ev.id, 'background', `background task #${ev.id}`, ''); const p = c.querySelector('pre'); p.textContent += ev.delta; p.scrollTop = p.scrollHeight; },
    'background.content'(ev) { const c = taskCard(ev.id, 'background', `background task #${ev.id}`, ''); c.querySelector('.result').textContent += ev.delta; },
    'background.tool_call'(ev) { const c = taskCard(ev.id, 'background', `background task #${ev.id}`, ''); c.querySelector('.tools').textContent += `→ ${ev.name}(${ev.arguments}) `; },
    'background.tool_result'(ev) { const c = taskCard(ev.id, 'background', `background task #${ev.id}`, ''); c.querySelector('.tools').textContent += `✓ ${Math.round(ev.ms)}ms  `; },
    'background.round'(ev) { const c = taskCard(ev.id, 'background', `background task #${ev.id}`, ''); c.querySelector('.status').textContent = `round ${ev.round + 1} · ${ev.tokens} tok · ${ev.tps} tok/s`; },
    'background.done'(ev) { const c = taskCard(ev.id, 'background', `background task #${ev.id}`, ''); c.querySelector('.result').textContent = ev.result; c.querySelector('details').open = false; setStatus(c, `done in ${fmtMs(ev.ms)} · waiting for a gap`, 'done'); },
    'timer.set'(ev) { const c = taskCard(ev.id, 'timer', `timer #${ev.id}`, `${ev.label} — ${ev.seconds}s`, true); c.querySelector('details').remove(); setStatus(c, `fires in ${ev.seconds}s`, 'running'); },
    'timer.fired'(ev) { const c = document.getElementById('task-' + ev.id); if (c) setStatus(c, 'fired · waiting for a gap', 'done'); },
    proactive(ev) { const c = document.getElementById('task-' + ev.id); if (c) setStatus(c, cfg.mode === 'duplex' ? 'handed to the duplex model' : 'delivered proactively', 'delivered'); },
    'duplex.status'(ev) {
      const pill = $('#pill-duplex');
      const st = { loading: 'loading its prompt…', ready: `ready (${fmtMs(ev.load_ms)}${ev.tool_outputs ? '' : ', no tool results'})`, error: 'error', closed: 'closed' }[ev.state] || ev.state;
      pill.classList.toggle('off', ev.state === 'error' || ev.state === 'closed');
      pill.title = `full-duplex speech model — ${st}${ev.error ? ': ' + ev.error : ''}`;
      if (cfg.mode === 'duplex') $('#llm-stat').textContent = `duplex ${st}`;
    },
    'duplex.utterance'(ev) {
      if (!ev.text) return;
      const li = document.createElement('li'); li.className = 'done'; li.innerHTML = `<span class="st"></span><span class="txt">${esc(ev.text)}</span><span class="m">duplex model</span>`;
      $('#chunks').appendChild(li); capChildren($('#chunks'), 80); li.scrollIntoView({ block: 'nearest' });
    },
    'duplex.yield'(ev) { if (curAssistant) { const c = document.createElement('span'); c.className = 'cut'; c.textContent = ev.kept_talking ? ' ⇄ kept talking over you' : ` ✂ yielded ${fmtMs(ev.ms)} after you cut in`; curAssistant.appendChild(c); } },
    log() {}, pong() {},
  };
  function onEvent(ev) {
    if (ev.type !== 'vad' && ev.type !== 'llm.content' && ev.type !== 'llm.reasoning' && ev.type !== 'background.reasoning' && ev.type !== 'background.content' && !(ev.type === 'asr.partial' && ev.duplex)) log(ev);
    const h = handlers[ev.type]; if (h) h(ev);
  }
  connect();
})();
