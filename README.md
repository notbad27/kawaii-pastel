# Kawaii Pastel Stable Diffusion Worker

A RunPod Serverless worker for generating images using the custom Kawaii Pastel Stable Diffusion model.

## Features

- Custom model loading (`nyl_kawaii_pastel.safetensors`)
- Automatic model loading with fallback methods
- Base64 image encoding for API responses
- Configurable generation parameters

## Quick Start

### 1. Build and Push Docker Image

**Windows (PowerShell):**
```powershell
.\build.ps1
```

**Linux/Mac:**
```bash
chmod +x build.sh
./build.sh
```

**Manual build:**
```bash
docker build --platform linux/amd64 --tag notbad27/kawaii-pastel-worker .
docker push notbad27/kawaii-pastel-worker:latest
```

### 2. Deploy on RunPod

1. Go to [RunPod Console - Serverless](https://www.console.runpod.io/serverless)
2. Click **New Endpoint** or edit existing endpoint
3. Use image: `docker.io/notbad27/kawaii-pastel-worker:latest`
4. Configure GPU (recommended: 24GB for Stable Diffusion)
5. Deploy!

### 3. Test Locally

```bash
python rp_handler.py
```

## API Usage

### Request Format

```json
{
    "input": {
        "prompt": "a cute kawaii pastel cat, pastel colors, soft lighting",
        "negative_prompt": "blurry, low quality, distorted",
        "num_inference_steps": 50,
        "guidance_scale": 7.5,
        "width": 512,
        "height": 512
    }
}
```

### Response Format

```json
{
    "image": "base64_encoded_image_string",
    "prompt": "your prompt",
    "status": "success"
}
```

## Files

- `rp_handler.py` - Main handler with model loading logic
- `Dockerfile` - Docker image configuration
- `requirements.txt` - Python dependencies
- `test_input.json` - Test input file
- `nyl_kawaii_pastel.safetensors` - Custom model file

## Model Loading

The handler supports two loading methods:
1. **Single file checkpoint** - If the safetensors file is a full checkpoint
2. **Base model + custom weights** - Loads base SD 1.5 and applies custom weights

The handler automatically tries both methods and uses whichever works.

