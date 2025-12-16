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

# Pre-download base Stable Diffusion model (3.4GB) to avoid runtime download
# This prevents "No space left on device" errors when workers start
# Note: This step requires ~4GB free space during build
RUN python3 -c "from diffusers import StableDiffusionPipeline; \
    import torch; \
    print('Pre-downloading base SD 1.5 model...'); \
    StableDiffusionPipeline.from_pretrained(\
        'runwayml/stable-diffusion-v1-5', \
        torch_dtype=torch.float16, \
        cache_dir='/root/.cache/huggingface'\
    ); \
    print('Base model cached successfully!')"

# Copy model file
COPY nyl_kawaii_pastel.safetensors /

# Copy your handler file
COPY rp_handler.py /

# Start the container
CMD ["python3", "-u", "rp_handler.py"]
