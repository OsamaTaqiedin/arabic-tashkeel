# RunPod Setup

This project is prepared for a persistent RunPod GPU pod with:

- repo clone at `/workspace/project`
- strict dataset at `/workspace/data/output_strict`
- checkpoints and logs at `/workspace/artifacts`

## Recommended pod

Use a persistent Community Cloud pod with one of:

- RTX 3090 24 GB
- RTX 4090 24 GB

Recommended disk sizing:

- 15 GB for repo, environments, and Python packages
- 2 GB for `output_strict`
- 5-20 GB for checkpoints, alternate runs, and logs
- practical minimum: 30 GB

## 1. Launch the pod

Choose a persistent pod, not ephemeral/serverless.

## 2. Clone the repo

```bash
mkdir -p /workspace
cd /workspace
git clone <YOUR_REPO_URL> project
cd /workspace/project
```

## 3. Install Python dependencies

Assuming the RunPod image already has Python 3:

```bash
cd /workspace/project
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 4. Install CUDA PyTorch

For Linux with CUDA 12.8 wheels:

```bash
python -m pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Verify:

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-gpu')"
```

## 5. Upload the prepared dataset once

Create the target directory:

```bash
mkdir -p /workspace/data/output_strict
```

Upload these local files into `/workspace/data/output_strict`:

- `train.jsonl`
- `validation.jsonl`
- `test.jsonl`
- `dataset_manifest.json`

You do not need to upload `rejected_samples.jsonl` for training.

## 6. Run validation checks

Environment check:

```bash
python --version
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Tiny smoke run without creating subset files:

```bash
cd /workspace/project
python train_tashkeel_seq2seq.py \
  --train-path /workspace/data/output_strict/train.jsonl \
  --validation-path /workspace/data/output_strict/validation.jsonl \
  --test-path /workspace/data/output_strict/test.jsonl \
  --checkpoint-dir /workspace/artifacts/gru_seq2seq_smoke \
  --device cuda \
  --use-amp \
  --limit-train-examples 100 \
  --limit-validation-examples 10 \
  --limit-test-examples 10 \
  --embedding-dim 16 \
  --encoder-hidden-size 16 \
  --decoder-hidden-size 32 \
  --batch-size 8 \
  --epochs 1 \
  --run-test
```

2k calibration run:

```bash
cd /workspace/project
python train_tashkeel_seq2seq.py \
  --train-path /workspace/data/output_strict/train.jsonl \
  --validation-path /workspace/data/output_strict/validation.jsonl \
  --test-path /workspace/data/output_strict/test.jsonl \
  --checkpoint-dir /workspace/artifacts/gru_seq2seq_2k \
  --device cuda \
  --use-amp \
  --limit-train-examples 2000 \
  --limit-validation-examples 200 \
  --limit-test-examples 200 \
  --embedding-dim 32 \
  --encoder-hidden-size 32 \
  --decoder-hidden-size 64 \
  --batch-size 8 \
  --epochs 1 \
  --run-test
```

## 7. Run training with the wrapper

Default wrapper:

```bash
cd /workspace/project
bash scripts/run_train.sh \
  --embedding-dim 32 \
  --encoder-hidden-size 32 \
  --decoder-hidden-size 64 \
  --batch-size 8 \
  --epochs 1
```

The wrapper automatically uses:

- `/workspace/data/output_strict`
- `/workspace/artifacts/gru_seq2seq_runpod`
- `--device cuda`
- `--use-amp`
- `--resume-from-latest`

## 8. Resume a run

If a pod stops or training is interrupted, rerun the same command with the same checkpoint directory.

Manual example:

```bash
python train_tashkeel_seq2seq.py \
  --train-path /workspace/data/output_strict/train.jsonl \
  --validation-path /workspace/data/output_strict/validation.jsonl \
  --test-path /workspace/data/output_strict/test.jsonl \
  --checkpoint-dir /workspace/artifacts/gru_seq2seq_runpod \
  --device cuda \
  --use-amp \
  --resume-from-latest
```

## 9. Download results

At minimum, export:

- `best.pt`
- `latest.pt`
- `vocab.json`
- `config.json`
- `history.json`

## Notes

- The current trainer writes epoch timing into `history.json` and stdout as `epoch_seconds`.
- Checkpoint selection is based on best validation diacritic CER.
- Mixed precision is enabled only when `--device cuda` and `--use-amp` are both set.
