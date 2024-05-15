#!/usr/bin/env bash

# Download and extract MuJoCo
mkdir -p ~/.mujoco
wget https://www.roboti.us/download/mujoco210-macos.zip -O mujoco210-macos.zip
unzip mujoco210-macos.zip -d ~/.mujoco
mv ~/.mujoco/mujoco210_macos ~/.mujoco/mujoco210
rm mujoco210-macos.zip

# Copy your license key to the right location
cp /path/to/your/mjkey.txt ~/.mujoco

# Install dependencies
brew install glfw glew

# Set environment variables
echo 'export MUJOCO_PY_MUJOCO_PATH="$HOME/.mujoco/mujoco210"' >> ~/.zshrc
echo 'export LD_LIBRARY_PATH="$HOME/.mujoco/mujoco210/bin:$LD_LIBRARY_PATH"' >> ~/.zshrc
source ~/.zshrc

# Install mujoco-py
pip install mujoco-py