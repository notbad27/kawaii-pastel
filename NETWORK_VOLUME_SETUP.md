# RunPod Network Volume Setup Guide

## Bakit kailangan ng Network Volume?

Kung hindi mo ma-build ang Docker image na may pre-downloaded model (dahil sa space issues), pwede mong gamitin ang **Network Volume** approach:

1. **Build Docker image WITHOUT pre-download** (maliit lang, mabilis)
2. **Setup Network Volume sa RunPod** (may malaking storage)
3. **Download model sa Network Volume sa first run** (cached na after)

## Step 1: Update Dockerfile (Remove Pre-download)

```dockerfile
# Remove the pre-download step para maliit ang image
# Model will be downloaded to network volume at runtime
```

## Step 2: Setup Network Volume sa RunPod

### Option A: Via RunPod Console (Easiest)

1. **Go to RunPod Console:**
   - https://www.runpod.io/console/storage

2. **Create Network Volume:**
   - Click **"Create Volume"**
   - Name: `kawaii-pastel-models` (o kahit anong gusto mo)
   - Size: **10GB** (minimum, para sa 3.4GB model + cache)
   - Click **"Create"**

3. **Attach Volume to Endpoint:**
   - Go to: https://www.runpod.io/console/serverless
   - Click your endpoint o create new one
   - Scroll down to **"Volume"** section
   - Select: `kawaii-pastel-models`
   - Mount path: `/workspace` (default, o `/runpod-volume`)
   - Save/Deploy

### Option B: Via RunPod CLI

```bash
# Install RunPod CLI (if not installed)
pip install runpod

# Login
runpod login

# Create volume
runpod volume create --name kawaii-pastel-models --size 10

# Get volume ID
runpod volume list

# Update endpoint with volume
# (Use RunPod console for this, mas madali)
```

## Step 3: Update Handler Code

Ang `rp_handler.py` mo ay may network volume detection na! Check mo:

- Line 35-48: Checks for `/runpod-volume`, `/workspace`, `/mnt/workspace`
- Line 50-56: Uses network volume for HuggingFace cache
- Line 83-89: Fallback to network volume if no pre-built cache

**Kung may network volume:**
- Model will download to: `/workspace/.cache/huggingface/` (o `/runpod-volume/.cache/huggingface/`)
- First run: Mag-download (5-10 minutes)
- Next runs: Instant (cached na!)

## Step 4: Update Dockerfile (Remove Pre-download)

Kung gusto mong gamitin ang network volume approach, tanggalin mo yung pre-download step:

```dockerfile
# Remove this:
# RUN python3 -c "from diffusers import StableDiffusionPipeline; ..."

# Keep this:
COPY nyl_kawaii_pastel.safetensors /
COPY rp_handler.py /
```

## Step 5: Build & Deploy

```bash
# Build (maliit na lang, walang model)
docker build --platform linux/amd64 --tag notbad27/kawaii-pastel-worker:latest .

# Push
docker push notbad27/kawaii-pastel-worker:latest

# Deploy sa RunPod with network volume attached
```

## Testing

1. **First Request:**
   - Worker starts
   - Downloads model to network volume (5-10 min)
   - Caches it
   - Processes request

2. **Next Requests:**
   - Worker starts
   - Loads from network volume cache (instant!)
   - Processes request

## Pros & Cons

### Network Volume Approach:
✅ **Pros:**
- Small Docker image (fast build, fast push)
- Works with GitHub Actions (no space issues)
- Model cached sa network volume (persistent)
- Multiple workers can share same cache

❌ **Cons:**
- First request takes longer (download model)
- Need to setup network volume
- Costs extra (network volume storage)

### Pre-download Approach:
✅ **Pros:**
- Fast first request (model already in image)
- No network volume needed
- Simpler setup

❌ **Cons:**
- Large Docker image (slow build, slow push)
- GitHub Actions can't build (no space)
- Need to build locally

## Recommendation

**Kung may space ka locally:** Use pre-download approach, build locally
**Kung walang space:** Use network volume approach, build sa GitHub Actions

---

## Quick Commands

```bash
# Check if volume is mounted
docker exec <container> ls -la /workspace

# Check cache location
docker exec <container> ls -la /workspace/.cache/huggingface/

# Check disk space
docker exec <container> df -h
```

