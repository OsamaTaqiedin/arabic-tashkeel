# FastAPI Inference App

This subdirectory contains a self-contained FastAPI backend and browser UI for
running Arabic diacritization inference from a saved checkpoint artifact
directory.

## Supported checkpoint formats

- Attention model artifacts produced by `train_tashkeel_seq2seq.py`
- Simple model artifacts produced by `train_simple_char_seq2seq.py`

The artifact directory must contain:

- `best.pt` or `latest.pt`
- `config.json`
- `vocab.json`

## Install

```bash
python -m pip install -r inference_app/requirements.txt
```

This app includes its own checkpoint loader, vocabulary code, and inference
model definitions. It does not need to import the training package at runtime.

## Run

```bash
python inference_app/app.py --artifact-dir "artifacts/runpod_prep_smoke"
```

Then open:

- [http://127.0.0.1:8000](http://127.0.0.1:8000)

## Example API request

```bash
curl -X POST http://127.0.0.1:8000/api/predict \
  -H "Content-Type: application/json" \
  -d "{\"text\":\"هذا كتاب\"}"
```
