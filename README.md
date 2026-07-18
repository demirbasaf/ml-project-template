<!-- Replace {{PLACEHOLDERS}} and delete these HTML comments as you go. -->
# {{project-name}}

> One-line description: what this model does, in plain language.

A small ML project built to the "**repo = source of truth, website = presentation
layer**" pattern: the trained model lives here as a static artifact, and any
consumer (a browser demo, or a personal site) runs it **client-side**, no server,
no inference cost.

> **Live demo:** [`web/`](web/) locally · embedded on [demirbasaf.dev](https://demirbasaf.dev)

## Sample output

```
{{paste a few real, representative outputs here}}
```

## How it works

{{2, 4 sentences + a tiny diagram of the forward pass. Keep it readable.}}

## Run it yourself

```bash
pip install -r requirements.txt

python train/train.py            # → train/checkpoint.pt   (seeded, reproducible)
python scripts/export_model.py   # → model/weights.json + model/meta.json
```

### Run the demo

```bash
python -m http.server 8000       # from the repo root
# open http://localhost:8000/web/
```

## Data

{{Where the data came from + license/provenance. If derived, ship the derived
form and .gitignore the raw source.}}

## Results

| split | metric |
|-------|--------|
| train | {{...}} |
| dev   | {{...}} |

## Repository layout

```
{{project-name}}/
├── README.md
├── data/                   # training data (or a note on how to get it)
├── train/                  # training code / notebook  → checkpoint
├── model/                  # ← the ARTIFACT (source of truth): meta.json + weights.json
├── scripts/export_model.py # trained model → model/ artifact
├── web/                    # self-contained browser demo (no build step)
│   ├── index.html
│   └── inference.js
├── requirements.txt
└── LICENSE
```

See **[`model/README.md`](model/README.md)** for the artifact convention, the
contract that lets any front-end consume any project here uniformly.

## Using this template

This is a **GitHub template repository**. Start a new project with:

```bash
gh repo create my-new-model --template demirbasaf/ml-project-template --public --clone
```

or click **“Use this template”** on GitHub. Then: drop your data in `data/`,
adapt `train/train.py`, run the two commands above, and fill in this README.

## License

MIT, see [LICENSE](LICENSE).
