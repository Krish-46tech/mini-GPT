# MiniGPT

> A small, readable GPT built from first principles in PyTorch.

MiniGPT is a character-level language model trained on the Tiny Shakespeare dataset. The entire learning loop lives in [`train.py`](train.py): data loading, tokenization, causal self-attention, Transformer blocks, optimization, checkpointing, and text generation.

The project is intentionally compact. It is designed to make the core mechanics of a GPT-style model easy to inspect, modify, and learn from.

## What It Does

Running the training script will:

1. Download `input.txt` from the Tiny Shakespeare dataset if the file is missing.
2. Build a character vocabulary and split the corpus into training and validation data.
3. Train a causal Transformer with next-character prediction.
4. Report training and validation loss during training.
5. Save model weights to `mini_gpt.pt`.
6. Generate and print 500 characters of sample text.

## Architecture

MiniGPT uses the following configuration by default:

| Component | Value |
| --- | ---: |
| Tokenization | Character-level |
| Context length | 128 characters |
| Embedding size | 256 |
| Attention heads | 4 |
| Transformer blocks | 4 |
| Dropout | 0.2 |
| Training steps | 3,000 |
| Optimizer | AdamW |
| Learning rate | 3e-4 |
| Random seed | 1337 |

Each Transformer block combines:

- Layer-normalized masked multi-head self-attention
- A two-layer feed-forward network with ReLU
- Residual connections around both sublayers

The model predicts the next character at every position. During generation, it repeatedly samples the next character while keeping only the most recent 128 characters as context.

## Quickstart

### 1. Clone the repository

```bash
git clone https://github.com/Krish-46tech/mini-GPT.git
cd mini-GPT
```

### 2. Create an environment

Python 3.9 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install PyTorch using the command appropriate for your operating system and hardware from [pytorch.org](https://pytorch.org/get-started/locally/). For a CPU-only macOS setup, this is typically:

```bash
pip install torch
```

### 3. Train and generate text

```bash
python train.py
```

The first run downloads the dataset when `input.txt` is not present. A trained checkpoint is written to:

```text
mini_gpt.pt
```

The script automatically uses CUDA when available and otherwise runs on the CPU.

## Adjusting Training

The main hyperparameters are defined near the top of [`train.py`](train.py). For a faster experiment on a laptop, reduce the training steps and evaluation work:

```python
max_iters = 500
eval_iters = 50
```

For better generations, increase `max_iters` and consider changing the model size. Larger values for `n_embd`, `n_head`, and `n_layer` require more memory and training time.

The dataset can also be replaced by putting another UTF-8 text corpus in `input.txt`. Since the tokenizer is built from the characters found in that file, the model will automatically adapt to the new vocabulary when trained from scratch.

## Project Layout

```text
.
├── train.py       # Model, tokenizer, training loop, and generation
├── train.ipynb    # Notebook version for interactive exploration
├── input.txt      # Training corpus; downloaded if absent
├── mini_gpt.pt    # Saved model weights
└── LICENSE
```

## Learning Goals

This repository is a useful starting point for exploring:

- Integer token IDs and embedding tables
- Positional embeddings
- Causal attention masks
- Multi-head self-attention
- Residual connections and layer normalization
- Cross-entropy next-token prediction
- Autoregressive sampling

It intentionally omits production features such as distributed training, mixed precision, experiment tracking, temperature/top-k controls, and a standalone checkpoint-loading script.

## Credits

The implementation follows the ideas demonstrated in Andrej Karpathy's *Let's build GPT from scratch* series and uses the Tiny Shakespeare corpus published in the `karpathy/char-rnn` repository.

## License

See [`LICENSE`](LICENSE) for the terms covering this project.