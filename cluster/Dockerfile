FROM pytorch/pytorch:2.9.0-cuda13.0-cudnn9-devel

# System deps for decord (video decoding)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libavformat-dev \
    libavcodec-dev \
    libavutil-dev \
    libswscale-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Python deps for vjepa2 training pipeline
RUN pip install --no-cache-dir \
    webdataset \
    decord \
    einops \
    timm \
    transformers \
    huggingface_hub \
    pandas \
    pyarrow \
    pyyaml \
    scipy \
    scikit-image \
    opencv-python-headless \
    iopath \
    beartype \
    psutil \
    ftfy \
    python-box \
    braceexpand \
    fire

WORKDIR /workspace
