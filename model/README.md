# The model artifact (the source of truth)

This folder is the **contract** between training (Python) and inference
(the browser). Any consumer, the demo in [`../web`](../web), or the personal
website, reads these files with a plain `fetch()` and needs **no per-model
code**. Keep this convention stable and every project here stays interchangeable.

A model is one folder with two files:

```
model/
├── meta.json      # everything needed to render + run the model, EXCEPT the weights
└── weights.json   # the weights (for tiny models), OR, model.onnx (for bigger ones)
```

## `meta.json`

```jsonc
{
  "name": "turkish-makemore",     // folder/id; must be URL-safe
  "title": "Turkish name generator",
  "description": "One line shown in the demo UI.",

  "backend": "js",                // "js" (tiny, hand-written forward pass) | "onnx"
  "arch": "mlp",                  // js only: "bigram" | "mlp"

  "vocab": [".", "a", "b", ...],  // index → token. vocab[0] MUST be the "." start/end token
  "block_size": 3,                // how many previous tokens the model conditions on

  "embedding_dim": 10,            // js + arch "mlp" only
  "hidden_size": 64,              // js + arch "mlp" only

  "sampling": {                   // defaults the UI starts from (user can override)
    "temperature": 1.0,
    "max_new_tokens": 20,
    "top_k": null                 // null = disabled
  }
}
```

**Invariants** (a consumer relies on all of these):
- `vocab[0]` is the boundary token `"."`, generation starts with a full window
  of it and stops when the model emits it.
- `vocab` order **is** the label→index map; `weights.json` indexes into it.
- For `backend: "js"`, `arch` decides the forward pass (`bigram` = a `V×V`
  lookup table; `mlp` = embed → `tanh` hidden → project).

## `weights.json` (js backend)

Every parameter tensor as a flat, **row-major** `data` array plus its `shape`:

```jsonc
{
  "format": "flat",
  "params": {
    "C":  { "shape": [30, 10], "data": [ ... ] },   // element [i,k] = data[i*10 + k]
    "W1": { "shape": [30, 64], "data": [ ... ] },
    "b1": { "shape": [64],     "data": [ ... ] },
    "W2": { "shape": [64, 30], "data": [ ... ] },
    "b2": { "shape": [30],     "data": [ ... ] }
  }
}
```

The **parameter names and shapes are the contract** with the forward pass in
[`../web/inference.js`](../web/inference.js) (and the website's `js-backend`):

| arch | required params |
|------|-----------------|
| `bigram` | `W` `[V, V]` (rows are next-token logits) |
| `mlp` | `C` `[V, emb]`, `W1` `[block·emb, hidden]`, `b1`, `W2` `[hidden, V]`, `b2` |

Row-major means `W[i][o] = data[i * outDim + o]`. `scripts/export_model.py`
writes exactly this; don't hand-edit it.

## `model.onnx` (onnx backend)

For models too big for a hand-written forward pass, export an ONNX graph
instead of `weights.json` and set `"backend": "onnx"` in `meta.json`. Contract:
input `int64 [1, block_size]` named `input`; output `float [1, vocab]` named
`logits`. `onnxruntime-web` runs it; sampling stays identical to the js path.

## Size guidance

| params | ship as | why |
|--------|---------|-----|
| ≲ 100k | `weights.json` | JSON is tiny; forward pass is a few JS loops, zero deps |
| ≳ 100k | `model.onnx` | avoids huge JSON; runtime loaded lazily, only when used |
