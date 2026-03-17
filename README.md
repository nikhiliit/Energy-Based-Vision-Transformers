# Energy-Based Vision Transformers

This repository contains the implementation and blog post for **Energy-Based Vision Transformers (EBT)**, a project that explores the intersection of Energy-Based Models (EBMs) and Vision Transformers (ViTs) for fine-grained image classification on the Oxford-IIIT Pets dataset.

## 🚀 [View the Live Blog Post](https://nikhiliit.github.io/Energy-Based-Vision-Transformers/)

---

## Project Overview

Most classifiers output a probability distribution directly. Energy-Based Models (EBMs) sidestep this by learning a scalar *energy function* $E(x, y)$ instead. This project demonstrates how a Vision Transformer can be re-interpreted as an engine for energy minimization, providing:

1.  **Semantic Geometry**: Learning distinct low-energy basins for 37 pet breeds.
2.  **Calibrated Confidence**: Using the energy gap between predictions as an inherent uncertainty metric.
3.  **Out-of-Distribution Detection**: Identifying non-pet images (like noise or vehicles) via free energy scores without additional training.

## Repository Structure

- `index.html`: The main blog post (HTML/CSS/JS).
- `assets/`: 
    - `img/`: High-resolution figures, attention maps, and confusion matrices.
    - `gifs/`: 3D energy surface evolutions and training animations.
- `EBT/`: The core Python engine.
    - `main.py`: CLI entry point for training and analytics.
    - `ebt/`: Module containing model definitions, loss functions, and training logic.
    - `data/`: Data loading and augmentation pipelines for Oxford-IIIT Pets.

## Quick Start (Python)

Detailed installation and usage instructions are located in [EBT/Readme.md](EBT/Readme.md).

```bash
# Clone the repository
git clone https://github.com/nikhiliit/Energy-Based-Vision-Transformers.git
cd Energy-Based-Vision-Transformers/EBT

# Install dependencies
pip install -r requirements.txt

# Run analytics (requires a trained checkpoint)
python main.py analytics --checkpoint path/to/model.pt --output_dir ../assets/img
```

## References

- LeCun et al., *[A Tutorial on Energy-Based Learning](http://yann.lecun.com/exdb/publis/pdf/lecun-06.pdf)* (2006)
- Ramsauer et al., *[Hopfield Networks is All You Need](https://arxiv.org/abs/2008.02217)* (2020)
- Dosovitskiy et al., *[An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)* (2020)
- Grathwohl et al., *[Your Classifier is Secretly an Energy Based Model](https://arxiv.org/abs/1912.03263)* (2019)
