# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a machine learning research project that fine-tunes **TurkishBERTweet** (`VRLLab/TurkishBERTweet`) — a RoBERTa-based model pre-trained on Turkish tweets — to detect cyberbullying. The study compares model performance across three temporal datasets: pre-COVID, post-COVID, and combined, examining how cyberbullying patterns changed over the pandemic period.

## Running the Notebook

All project code lives in a single Jupyter notebook: `TurkishBERTweet.ipynb`

```bash
# Launch Jupyter
jupyter notebook TurkishBERTweet.ipynb

# Or via JupyterLab
jupyter lab TurkishBERTweet.ipynb
```

The notebook runs three experiments sequentially (pre-COVID → post-COVID → combined). Each experiment fine-tunes a fresh model instance and reports Accuracy, Precision, Recall, and F1. GPU memory is explicitly released after each run.

## Dependencies

No `requirements.txt` exists. Install manually:

```bash
pip install torch transformers pandas numpy scikit-learn openpyxl
```

- PyTorch with CUDA is strongly recommended; the code auto-detects GPU availability.
- The `openpyxl` engine is required for reading `.xlsx` dataset files.

## Architecture

The notebook is structured as a single pipeline:

1. **`CyberbullyingDataset`** — PyTorch `Dataset` subclass that tokenizes raw tweet text using the TurkishBERTweet tokenizer (`max_len=256`, padding, attention masks) and returns label tensors.

2. **`train_epoch()`** — One epoch of fine-tuning with AdamW optimizer and a linear warmup scheduler. Gradient clipping is applied (`max_norm=1.0`).

3. **`eval_model()`** — Evaluates on a DataLoader and returns macro-averaged precision, recall, F1, accuracy, and mean loss using `sklearn.metrics`.

4. **`run_experiment(path)`** — Orchestrates a full fine-tuning run: loads an Excel file, performs an 80/20 train-test split, initializes a fresh `BertForSequenceClassification` head on top of TurkishBERTweet (2 labels), trains for 4 epochs with a linear schedule (10% warmup), checkpoints the best model by validation F1, evaluates on the test split, and clears GPU memory.

## Key Hyperparameters (paper-matched)

| Parameter | Value |
|---|---|
| `EPOCHS` | 4 |
| `LEARNING_RATE` | 2e-5 |
| `BATCH_SIZE` | 16 |
| `MAX_LEN` | 256 |
| `RANDOM_STATE` | 42 |
| Weight decay | 0.01 (non-bias/LayerNorm params only) |
| LR warmup | 10% of total training steps |

## Data

Excel files in `data/`:
- `covidoncesi.xlsx` — pre-COVID tweets
- `covidsonrasi.xlsx` — post-COVID tweets
- `covidoncesivesonrasi.xlsx` — combined dataset

Each file contains Turkish tweet text and binary cyberbullying labels. The data is expected in the first two columns (text, label).

## Known Results (from last run)

| Dataset | Accuracy | F1 |
|---|---|---|
| Pre-COVID | 0.7911 | 0.7302 |
| Post-COVID | 0.6424 | 0.5986 |
| Combined | 0.7040 | 0.6548 |
