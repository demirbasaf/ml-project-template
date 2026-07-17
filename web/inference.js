// ---------------------------------------------------------------------------
// inference.js - the whole model, running in your browser, in plain JavaScript.
//
// This is the same MLP forward pass as train/train.py, re-implemented with a
// few array loops so there is no framework and nothing to hide. Read it top to
// bottom and you can see exactly how a character-level language model samples
// text: embed a window of previous characters, one tanh hidden layer, project
// to next-character scores, turn those into probabilities, draw one, repeat.
//
// It loads two static files (no server, no API):
//   model/meta.json     vocab, block_size, hyperparams, sampling defaults
//   model/weights.json  every parameter as { shape, data(flat, row-major) }
// ---------------------------------------------------------------------------

/** Load the artifact (meta + weights) from a folder of static JSON files. */
export async function loadModel(baseUrl = '../model') {
  const [meta, weights] = await Promise.all([
    fetch(`${baseUrl}/meta.json`).then((r) => r.json()),
    fetch(`${baseUrl}/weights.json`).then((r) => r.json()),
  ]);
  return { meta, params: weights.params ?? weights };
}

/** y = x · W + b.  W is [inDim, outDim] stored row-major, so W[i,o] = data[i*outDim + o]. */
function linear(x, W, b) {
  const [inDim, outDim] = W.shape;
  const y = new Array(outDim);
  for (let o = 0; o < outDim; o++) {
    let s = b ? b.data[o] : 0;
    for (let i = 0; i < inDim; i++) s += x[i] * W.data[i * outDim + o];
    y[o] = s;
  }
  return y;
}

/**
 * The forward pass: context (array of token indices) -> logits over next token.
 * Mirrors train.py exactly:  h = tanh(concat(emb) · W1 + b1);  logits = h · W2 + b2.
 */
export function forward(context, meta, params) {
  const emb = meta.embedding_dim;
  const { C, W1, b1, W2, b2 } = params;
  // Look up each context token's embedding row and concatenate them.
  const x = [];
  for (const t of context) {
    for (let k = 0; k < emb; k++) x.push(C.data[t * emb + k]);
  }
  const h = linear(x, W1, b1).map(Math.tanh);
  return linear(h, W2, b2);
}

/** A tiny seeded PRNG so "reproducible with a fixed seed" works. */
export function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** logits -> one sampled index. temperature sharpens/flattens; optional top-k. */
export function sample(logits, { temperature = 1.0, topK = null, rng = Math.random }) {
  const n = logits.length;
  const T = Math.max(1e-6, temperature);
  const scaled = logits.map((v) => v / T);

  let pool = Array.from({ length: n }, (_, i) => i);
  if (topK && topK > 0 && topK < n) {
    pool = pool.sort((a, b) => scaled[b] - scaled[a]).slice(0, topK);
  }
  let max = -Infinity;
  for (const i of pool) if (scaled[i] > max) max = scaled[i];
  let sum = 0;
  const probs = new Float64Array(n);
  for (const i of pool) {
    const e = Math.exp(scaled[i] - max);
    probs[i] = e;
    sum += e;
  }
  const r = rng() * sum;
  let acc = 0;
  for (const i of pool) {
    acc += probs[i];
    if (r <= acc) return i;
  }
  return pool[pool.length - 1];
}

/**
 * Generate one name. Start with a full window of the '.' boundary token
 * (index 0), then autoregressively sample until the model emits '.' again.
 */
export function generate(meta, params, { temperature, topK = null, seed = null } = {}) {
  const { vocab, block_size } = meta;
  const T = temperature ?? meta.sampling.temperature;
  const k = topK ?? meta.sampling.top_k ?? null;
  const rng = seed == null ? Math.random : mulberry32(seed);

  let context = new Array(block_size).fill(0); // 0 === '.' boundary, by convention
  let out = '';
  for (let step = 0; step < meta.sampling.max_new_tokens; step++) {
    const next = sample(forward(context, meta, params), { temperature: T, topK: k, rng });
    if (next === 0) break; // model chose to end the name
    out += vocab[next];
    context = [...context.slice(1), next];
  }
  return out;
}
