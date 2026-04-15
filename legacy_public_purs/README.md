# Legacy Public PURS (Kaggle)

This folder provides an isolated, public-style PURS pipeline so you can mimic the original training loop while keeping your current PyTorch code untouched.

## What this reproduces

- Public-style model architecture and training protocol (TF v1 graph mode semantics)
- Public metrics and thresholding (`AUC`, `HitRate`, `Coverage`, `Unexpectedness`)
- Public input schema: `utdid,vdo_id,click,hour`
- Your environment parameters from YAML config:
  - `model.epochs`
  - `model.batch_size`
  - `model.learning_rate`
  - `model.history_length`
  - `model.gru_hidden_dim`
  - `experiment.seed`
  - `data.positive_rating_threshold`
  - `data.data_subset`

## Files

- `model_public_tf_compat.py`: Public model ported to `tf.compat.v1`
- `train_public_style.py`: End-to-end trainer + data conversion
- `requirements-kaggle-legacy.txt`: Isolated dependency set for Kaggle

## Kaggle run steps

1. Open a fresh Kaggle notebook (new session recommended).
2. Clone or upload this repository.
3. Install legacy deps:

```bash
pip install -q -r legacy_public_purs/requirements-kaggle-legacy.txt
```

This requirement set is pinned for modern Kaggle Python (3.12-compatible TensorFlow).
If a preinstalled JAX package causes an ml_dtypes conflict, the trainer auto-removes JAX/JAXLIB and retries TensorFlow import.
The legacy model uses a GRUCell-based RNN path to avoid CuDNN-only GRU kernel issues on mixed Kaggle GPU library stacks.

4. Run training using your existing config:

```bash
python legacy_public_purs/train_public_style.py \
  --config config/kaggle_config.yaml \
  --output-dir /kaggle/working/legacy_public_purs
```

Optional flags:

- `--raw-dir data/raw/ml-1m` when you already copied files into the repo workspace.
- `--raw-dir /kaggle/input/movielens-1m` if dataset path differs.
- `--dataset-name movielens-1m` or `movielens-32m`.
- `--data-path /path/to/test.txt` to skip auto-conversion.

If `--raw-dir` is omitted, the script also performs a recursive search across common
Kaggle and workspace directories for the ratings file.

## Outputs

- Generated public-format file: `<output-dir>/test.txt` (if auto-built)
- Training summary: `<output-dir>/metrics.json`

## Notes

- This is intentionally separate from your modern pipeline to avoid dependency conflicts.
- Exact numerical parity with the original paper/code can still vary by TF version, hardware, and dataset split nuances.
