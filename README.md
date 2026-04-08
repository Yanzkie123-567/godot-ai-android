# godot-ai-android

A custom transformer-based AI trained on Godot 4 Q&A data. Ask it how to make a player jump, hide a node, save game data — it answers with GDScript.

## Architecture

`godot_ai_final.py` implements `YourGameAI`, a GPT-2-family decoder trained from scratch on Godot scripting questions and answers.

| Component | Detail |
|---|---|
| Attention | Multi-head causal self-attention via `F.scaled_dot_product_attention` (Flash Attention on GPU) |
| Feed-forward | SwiGLU (gate × up projection, same as LLaMA/Mistral) |
| Weight tying | Token embedding matrix shared with output head (GPT-2 style) |
| Tokenizer | HuggingFace BPE (`PreTrainedTokenizerFast`) — handles GDScript identifiers well |
| Sampling | Top-K + Top-P (nucleus) with repetition penalty and NaN guard |

Default hyperparameters:
- `embed_dim = 512`, `num_layers = 6`, `num_heads = 8`, `ff_dim = 2048`
- `max_length = 256` tokens, `vocab_size = 8000` BPE subwords
- Training: AdamW + OneCycleLR + label smoothing 0.1 + AMP on GPU

## Files

| File | Description |
|---|---|
| `godot_ai_final.py` | Full training notebook (all 13 cells merged into one `.py`) |
| `godot_ultimate_v3.json` | Primary training dataset — Godot 4 Q&A pairs |
| `godot_dataset.json` | Supplementary Q&A dataset |
| `godot_dataset_rich_v2.json` | Enriched Q&A dataset |
| `myaimodel.ipynb` | Original Jupyter notebook |

## Quick Start (Kaggle)

1. Upload `godot_ultimate_v3.json` (or `godot_dataset.json`) as a Kaggle Dataset via **+ Add Data → Upload**.
2. In Notebook Settings set **Accelerator → GPU T4**.
3. Set the dataset path in `godot_ai_final.py`:
   ```python
   JSON_PATH = "/kaggle/input/<your-dataset-name>/godot_dataset.json"
   ```
4. Run all cells top-to-bottom. Wait for each `✅` before proceeding.

Outputs are saved to `/kaggle/working/GodotAI/`:
- `model/` — final model weights + tokenizer
- `checkpoints/` — best checkpoint + per-30-epoch snapshots
- `training_curve.png` — loss plot

## Making the AI Smarter

**Train longer:**
```python
history = train(your_ai, dataloader, tokenizer, num_epochs=200)
```

**Bigger model:**
```python
your_ai = YourGameAI(
    vocab_size=len(tokenizer),
    embed_dim=768,   # was 512
    num_layers=8,    # was 6
    num_heads=12,    # 768/12 = 64 ✓
    ff_dim=3072,     # 4 × 768
).to(DEVICE)
```

**Enable `torch.compile` (free ~25% speedup on PyTorch 2.0+):**
```python
USE_COMPILE = True  # in Cell 7
```

**Resume from checkpoint:**
```python
your_ai = load_checkpoint(f"{SAVE}/checkpoints/best_model.pt")
history = train(your_ai, dataloader, tokenizer, num_epochs=100)
```

**Loss targets:**
- `< 3.0` — model is learning
- `< 2.0` — answers start making sense
- `< 1.0` — mostly correct answers (aim here)
- `< 0.5` — well-trained model

## Chat Mode

`ChatSession` keeps conversation history (context memory up to 300 tokens):

```python
session = ChatSession(your_ai, tokenizer)
session.chat("how do i make a Sprite2D?")
session.chat("how do i hide it?")   # remembers Sprite2D context
session.chat("how do i delete it?")
session.reset()                      # clear memory
```

One-shot generation:
```python
your_ai.generate(
    tokenizer,
    "User: how do i make an enemy follow the player Assistant:",
    max_new_tokens=60,
    temperature=0.6,
)
```

## GitHub Actions — Claude Code

This repo includes a GitHub Actions workflow (`.github/workflows/`) that activates Claude Code when `@claude` appears in:
- Issue bodies or titles
- Issue comments
- Pull request review comments
- Pull request reviews

To enable it, add `ANTHROPIC_API_KEY` as a repository secret under **Settings → Secrets and variables → Actions**.

## Requirements

```
torch >= 2.0
transformers
tokenizers
numpy
matplotlib
```

Install:
```bash
pip install torch transformers tokenizers numpy matplotlib
```
