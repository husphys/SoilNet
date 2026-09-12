# Reproduction environment

Use `soilnet.yml` to create the Python environment and
`requirements-lock.txt` to install the fully pinned Python package set.

The frozen held-out evaluation records Python 3.11.15, PyTorch 2.6.0+cu118,
torchvision 0.21.0+cu118, timm 1.0.29, NumPy 2.4.6, and CUDA 11.8. The lock also
pins pandas 3.0.5 and scikit-learn 1.9.0. This is a tested reproduction
environment; cross-platform/GPU bitwise identity is not claimed.
