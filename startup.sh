#!/bin/bash

# Clone the GitHub repo (no login needed for public repos)
git clone https://github.com/MocanuPaulC/DL-assignment.git
# Install required Python packages
python3.11 -m pip install --upgrade pip
python3 -m pip install "tensorflow[and-cuda]" pandas optuna kagglehub
git config --global user.name "Paul"
git config --global user.email "mocanupaulc@gmail.com"

token=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

pip install \
  --extra-index-url https://pypi.nvidia.com \
  tensorrt-bindings==8.6.1 \
  tensorrt-libs==8.6.1 \
  tensorflow[and-cuda]==2.15.0 pandas matplotlib scikit-learn