#!/bin/bash

# Clone the GitHub repo (no login needed for public repos)
git clone https://github.com/MocanuPaulC/DL-assignment.git
# Install required Python packages
python3.11 -m pip install --upgrade pip
python3.11 -m pip install "tensorflow[and-cuda]" pandas optuna kagglehub
git config --global user.name "Paul"
git config --global user.email "mocanupaulc@gmail.com"