# Energy-Based Transformer (EBT)

A PyTorch implementation of an Energy-Based Transformer for image classification, trained on Oxford-IIIT Pets (37 breeds). This accompanies a blog post walking through the theory, architecture, and results.

---

## What is an Energy-Based Model?

Most classifiers output a probability distribution directly — they are *generative* in the sense that they must normalize over all possible outputs. Energy-Based Models (EBMs) sidestep this by learning a scalar *energy function* `E(x, y)` instead of a probability. Lower energy means higher compatibility between input `x` and label `y`.

The probability is recovered via the Boltzmann distribution:

$$P(y \mid x) = \frac{e^{-E(x, y)}}{\sum_{y'} e^{-E(x, y')}}$$

The denominator is the partition function `Z`. For continuous or high-dimensional outputs this integral is intractable. For finite discrete classification it is just a sum over classes — which makes EBMs practical here and gives us the OOD detection property described below.

## Why Transformers?

Transformers and EBMs connect at the level of their internal computations. Ramsauer et al. (2020) showed that modern Hopfield Networks — a class of EBMs designed for associative memory — perform energy minimization via the update:

$$\sigma^{\text{new}} = \xi \cdot \text{softmax}(\beta \, \xi^T \sigma)$$

where `ξ` are stored patterns and `σ` is the query state. This is exactly one step of softmax self-attention. The attention scores `QK^T / √d` are a compatibility (negative energy) matrix — the softmax that follows is the Gibbs distribution over keys. A Transformer block is therefore already doing energy-based reasoning at each layer; the EBT just makes the final output of that process explicit.

## Architecture

The model follows the Vision Transformer (ViT) design. A 224×224 image is split into 16×16 patches, linearly projected to an embedding dimension, and processed by a stack of pre-norm Transformer blocks. The CLS token representation `h(x)` after the final block is passed through a linear head that outputs one scalar per class:

$$E(x, c) = -(W \, h(x) + b)_c$$

The negation is a convention: the head naturally produces logits, and we negate them so that the highest-scoring class has the *lowest* energy.

**Inference** selects the minimum-energy class:

$$\hat{y} = \arg\min_c \, E(x, c)$$

## Training Objective

The loss is the negative log-likelihood under the Gibbs distribution:

$$\mathcal{L} = E(x, y^+) + \log \sum_c e^{-E(x, c)}$$

The first term pulls the correct-class energy down. The second term (the log-partition) pushes all energies up. Their combination creates a margin: the correct class settles lower than the rest. This is numerically equivalent to cross-entropy applied to `-energies`, but the EBM framing makes two things explicit:

1. **Energy gap** — the difference `E(x, ŷ₂) - E(x, ŷ₁)` between the two lowest energies is a direct confidence measure, not a softmax probability.
2. **OOD detection** — the free energy score `-log Σ_c exp(-E(x, c))` is high for in-distribution inputs (the model concentrates energy on one class) and low for out-of-distribution inputs (energy is spread uniformly). No separate uncertainty head is needed.

## Project Structure

```
ebt/
├── data/
│   └── pipeline.py       Oxford Pets loading, train/val/test splits
├── model/
│   ├── transformer.py    PatchEmbedding, TransformerBlock, EBT, PretrainedEBT
│   └── loss.py           energy_loss — EBM NLL for classification
├── training/
│   └── engine.py         train_one_epoch, evaluate
└── plotting/
    ├── analysis.py       IEEE-style analysis plots
    └── surface_3d.py     3D energy landscape renderer
main.py                   CLI: train / analytics / plot_3d
```

## Usage

**Train from scratch:**
```bash
python main.py train --epochs 50 --size small --lr 3e-4 --device cuda
```

**Train with pretrained ViT-B/16 backbone:**
```bash
python main.py train --epochs 20 --pretrained --lr 1e-4 --device cuda
```

**Generate analysis plots** (energy distributions, attention maps, OOD detection, t-SNE, calibration):
```bash
python main.py analytics --checkpoint checkpoints/ebt_best.pt --output_dir assets/
```

**3D energy surface for a single class:**
```bash
python main.py plot_3d --checkpoint checkpoints/ebt_best.pt --class_idx 5 --style sample
```

## Model Configurations

| Size       | Params | Depth | Embed dim | Heads |
|------------|--------|-------|-----------|-------|
| tiny       | ~5M    | 4     | 256       | 4     |
| small      | ~15M   | 6     | 384       | 6     |
| base       | ~86M   | 12    | 768       | 12    |
| pretrained | ~86M   | 12    | 768       | 12    |

The `pretrained` option loads ImageNet ViT-B/16 weights and replaces the classification head with a randomly initialised energy head.

## References

- Ramsauer et al., *Hopfield Networks is All You Need* (2020)
- LeCun et al., *A Tutorial on Energy-Based Learning* (2006)
- Dosovitskiy et al., *An Image is Worth 16×16 Words* (2020)
- Grathwohl et al., *Your Classifier is Secretly an Energy Based Model* (2019)
