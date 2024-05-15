# Use an ARM-compatible Python base image
FROM python:3.8-slim-buster

# Set environment variables to ensure non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG C.UTF-8

# Install system dependencies including the required libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenmpi-dev \
    zlib1g-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libosmesa6-dev \
    libglew-dev \
    libgl1-mesa-dev \
    patchelf \
    ffmpeg \
    git \
    wget \
    unzip \
    software-properties-common \
    net-tools \
    vim \
    virtualenv \
    xpra \
    xserver-xorg-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install pip and upgrade it
RUN pip install --upgrade pip

# Install MuJoCo binaries and place them in the .mujoco/mujoco210 directory
RUN mkdir -p /root/.mujoco && \
    wget -O /root/.mujoco/mujoco210-linux-aarch64.tar.gz https://github.com/deepmind/mujoco/releases/download/2.1.1/mujoco-2.1.1-linux-aarch64.tar.gz && \
    mkdir -p /root/.mujoco/mujoco210 && \
    tar -zxvf /root/.mujoco/mujoco210-linux-aarch64.tar.gz -C /root/.mujoco/mujoco210 --strip-components=1 && \
    rm /root/.mujoco/mujoco210-linux-aarch64.tar.gz

# Create symbolic links for the libraries
RUN ln -s /root/.mujoco/mujoco210/lib/libmujoco.so /root/.mujoco/mujoco210/lib/libmujoco210.so && \
    ln -s /root/.mujoco/mujoco210/lib/libglewosmesa.so /root/.mujoco/mujoco210/lib/libglewosmesa.so
    
# Set MuJoCo environment variables
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin
ENV MUJOCO_PY_MUJOCO_PATH=/root/.mujoco/mujoco210

# Copy the requirements.txt file into the container
COPY requirements.txt /workspace/requirements.txt

# Install Python packages from requirements.txt
RUN pip install --upgrade pip && \
    pip install -r /workspace/requirements.txt

# Set up a working directory
WORKDIR /workspace

# Copy the current directory contents into the container at /workspace
COPY . /workspace

# Set the default command to run when starting the container
CMD ["bash"]
