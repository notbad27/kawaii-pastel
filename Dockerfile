FROM python:3.10-slim

WORKDIR /

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model file
# Note: Base Stable Diffusion model will be downloaded at runtime to network volume
# Make sure your RunPod endpoint has a network volume configured!
# See NETWORK_VOLUME_SETUP.md for instructions
COPY nyl_kawaii_pastel.safetensors /

# Copy your handler file
COPY rp_handler.py /

# Start the container
CMD ["python3", "-u", "rp_handler.py"]
