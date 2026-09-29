# ==============================================================================
# GeoDelta Backend Dockerfile (Python 3.11 + CUDA 12.1 + GDAL/Rasterio)
# Problem Statement: SIH26227 (Ministry of Defence, Government of India)
# ==============================================================================

FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Prevent interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies, Python 3.11, and GDAL
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    curl \
    git \
    build-essential \
    libpq-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && add-apt-repository ppa:ubuntugis/ubuntugis-unstable -y \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    gdal-bin \
    libgdal-dev \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set python3.11 as default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Setup GDAL compiler paths
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

WORKDIR /app

# Upgrade pip and install PyTorch with CUDA 12.1 first
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && python -m pip install --no-cache-dir \
    torch==2.2.2 \
    torchvision==0.17.2 \
    --index-url https://download.pytorch.org/whl/cu121

# Install GDAL Python bindings matching the system version
RUN GDAL_VERSION=$(gdal-config --version) \
    && python -m pip install --no-cache-dir "GDAL==${GDAL_VERSION}.*"

# Install remaining dependencies from requirements.txt
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir -r requirements.txt

# Copy source code into container
COPY . /app/

# Environment paths
ENV PYTHONPATH=/app
ENV MODEL_WEIGHTS_DIR=/models/weights

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
