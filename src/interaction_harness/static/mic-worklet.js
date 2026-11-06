// Captures mic audio, downsamples to 16 kHz, and posts int16 frames of 512 samples (32 ms).
class MicProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ratio = sampleRate / 16000;   // e.g. 48000/16000 = 3
    this.pos = 0;                      // fractional read position into the carry buffer
    this.carry = new Float32Array(0);
    this.out = new Int16Array(512);
    this.n = 0;
    this.rmsAcc = 0; this.rmsN = 0;
  }
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch) return true;
    // append to carry
    const buf = new Float32Array(this.carry.length + ch.length);
    buf.set(this.carry); buf.set(ch, this.carry.length);
    let i = this.pos;
    while (i + 1 < buf.length) {
      const i0 = Math.floor(i), frac = i - i0;
      const s = buf[i0] * (1 - frac) + buf[i0 + 1] * frac;
      const v = Math.max(-1, Math.min(1, s));
      this.out[this.n++] = v < 0 ? v * 32768 : v * 32767;
      this.rmsAcc += v * v; this.rmsN++;
      if (this.n === 512) {
        this.port.postMessage({ pcm: this.out.buffer.slice(0), rms: Math.sqrt(this.rmsAcc / this.rmsN) });
        this.n = 0; this.rmsAcc = 0; this.rmsN = 0;
      }
      i += this.ratio;
    }
    const keep = Math.floor(i);
    this.carry = buf.slice(keep);
    this.pos = i - keep;
    return true;
  }
}
registerProcessor('mic-processor', MicProcessor);
