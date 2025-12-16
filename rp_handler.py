import runpod
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from safetensors.torch import load_file
import base64
import io
from PIL import Image
import os

# Global variables to store the loaded models
pipe = None
img2img_pipe = None

# Configure HuggingFace cache location
# Priority: Network volume > Container disk (/tmp with more space)
print("=" * 50)
print("Checking available storage locations...")
print("=" * 50)

# Check network volume mount points
NETWORK_VOLUME_PATH = None
for path in ['/runpod-volume', '/workspace', '/mnt/workspace', '/workspace/.runpod']:
    if os.path.exists(path):
        # Check if writable and has space
        try:
            test_file = os.path.join(path, '.test_write')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            NETWORK_VOLUME_PATH = path
            print(f"✓ Found network volume at: {path}")
            break
        except:
            print(f"✗ {path} exists but not writable")
            continue

if NETWORK_VOLUME_PATH:
    # Use network volume for cache
    hf_cache_dir = os.path.join(NETWORK_VOLUME_PATH, '.cache', 'huggingface')
    os.makedirs(hf_cache_dir, exist_ok=True)
    os.environ['HF_HOME'] = hf_cache_dir
    os.environ['HUGGINGFACE_HUB_CACHE'] = os.path.join(hf_cache_dir, 'hub')
    print(f"✓ Using network volume for model cache: {hf_cache_dir}")
else:
    # Use /tmp which should have more space (container disk)
    hf_cache_dir = '/tmp/.cache/huggingface'
    os.makedirs(hf_cache_dir, exist_ok=True)
    os.environ['HF_HOME'] = hf_cache_dir
    os.environ['HUGGINGFACE_HUB_CACHE'] = os.path.join(hf_cache_dir, 'hub')
    print(f"✓ Using container disk (/tmp) for model cache: {hf_cache_dir}")
    print("  Note: Cache will be lost on container restart")

# Print disk space info
import shutil
total, used, free = shutil.disk_usage(hf_cache_dir)
print(f"Disk space at {hf_cache_dir}:")
print(f"  Total: {total / (1024**3):.2f} GB")
print(f"  Used: {used / (1024**3):.2f} GB")
print(f"  Free: {free / (1024**3):.2f} GB")
print("=" * 50)

def load_model():
    """Load the Stable Diffusion model once at startup"""
    global pipe, img2img_pipe
    if pipe is None:
        print("Loading custom kawaii pastel model...")
        model_path = "/nyl_kawaii_pastel.safetensors"
        
        # Check if model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        
        print(f"Found model file at: {model_path}")
        
        # Skip single file method - safetensors is not a full checkpoint
        # Go directly to base model + custom weights approach
        print("Loading base model and applying custom weights...")
        try:
            # Load a base SD 1.5 model first
            base_model = "runwayml/stable-diffusion-v1-5"
            print(f"Loading base model: {base_model}")
            print("Note: This may take a few minutes on first run...")
            
            # Use cache_dir from environment if network volume is available
            cache_dir = os.environ.get('HUGGINGFACE_HUB_CACHE', None)
            
            try:
                # Try to load from cache first
                load_kwargs = {
                    'torch_dtype': torch.float16 if torch.cuda.is_available() else torch.float32,
                    'local_files_only': True  # Try cache first
                }
                if cache_dir:
                    load_kwargs['cache_dir'] = cache_dir
                
                pipe = StableDiffusionPipeline.from_pretrained(
                    base_model,
                    safety_checker=None,  # Disable safety checker
                    requires_safety_checker=False,  # Disable safety checker
                    **load_kwargs
                )
                print("✓ Loaded base model from cache!")
            except Exception as cache_error:
                print(f"Cache miss: {cache_error}")
                print("Attempting to download base model (requires ~3.4GB disk space)...")
                # If cache miss, try downloading (will save to network volume if available)
                load_kwargs = {
                    'torch_dtype': torch.float16 if torch.cuda.is_available() else torch.float32,
                    'local_files_only': False
                }
                if cache_dir:
                    load_kwargs['cache_dir'] = cache_dir
                    print(f"Downloading to network volume cache: {cache_dir}")
                
                pipe = StableDiffusionPipeline.from_pretrained(
                    base_model,
                    safety_checker=None,  # Disable safety checker
                    requires_safety_checker=False,  # Disable safety checker
                    **load_kwargs
                )
                print("✓ Base model downloaded and cached!")
            
            # Load LoRA weights from safetensors file
            print(f"Loading LoRA weights from: {model_path}")
            try:
                # Load LoRA weights: pass directory and weight_name
                # model_path is "/nyl_kawaii_pastel.safetensors"
                lora_dir = os.path.dirname(model_path)  # "/"
                lora_file = os.path.basename(model_path)  # "nyl_kawaii_pastel.safetensors"
                
                print(f"Loading LoRA from directory: {lora_dir}, file: {lora_file}")
                pipe.load_lora_weights(lora_dir, weight_name=lora_file)
                print("✓ LoRA weights loaded successfully!")
                
                # Fuse LoRA weights for better performance (optional but recommended)
                try:
                    pipe.fuse_lora()
                    print("✓ LoRA weights fused for optimal performance!")
                except Exception as fuse_error:
                    print(f"Note: Could not fuse LoRA weights: {fuse_error}")
                    print("Continuing without fusion (slightly slower but still works)")
                
            except Exception as lora_error:
                print(f"Error loading LoRA weights: {lora_error}")
                import traceback
                traceback.print_exc()
                print("Warning: Could not load LoRA weights. Using base model only.")
            
            # Explicitly disable safety checker (double check)
            pipe.safety_checker = None
            pipe.requires_safety_checker = False
            
            if torch.cuda.is_available():
                pipe = pipe.to("cuda")
            
            try:
                pipe.enable_xformers_memory_efficient_attention()
            except:
                print("xformers not available, using default attention")
            
            # Create img2img pipeline from the same model (LoRA already applied to components)
            print("Creating img2img pipeline...")
            img2img_pipe = StableDiffusionImg2ImgPipeline(
                vae=pipe.vae,
                text_encoder=pipe.text_encoder,
                tokenizer=pipe.tokenizer,
                unet=pipe.unet,  # UNet already has LoRA applied
                scheduler=pipe.scheduler,
                safety_checker=None,  # Disable safety checker
                requires_safety_checker=False,  # Disable safety checker
                feature_extractor=pipe.feature_extractor,
            )
            
            # Explicitly disable safety checker (double check)
            img2img_pipe.safety_checker = None
            img2img_pipe.requires_safety_checker = False
            
            # Note: LoRA weights are already applied since we're using the same UNet
            print("✓ Img2Img pipeline created with LoRA weights!")
            
            if torch.cuda.is_available():
                img2img_pipe = img2img_pipe.to("cuda")
            
            try:
                img2img_pipe.enable_xformers_memory_efficient_attention()
            except:
                pass
            
            print("✓ Model loaded successfully!")
            print("✓ Img2Img pipeline ready!")
            print("✓ Safety checker disabled!")
            
        except Exception as e2:
                print(f"Error loading model: {e2}")
                import traceback
                traceback.print_exc()
                raise e2
    
    return pipe, img2img_pipe

def handler(job):
    """
    This function processes incoming requests to generate images using the kawaii pastel model.
    
    Args:
        job (dict): Contains the input data and request metadata
        - input.prompt: Text prompt for image generation
        - input.image (optional): Base64-encoded input image for img2img mode
        - input.negative_prompt (optional): Negative prompt
        - input.num_inference_steps (optional): Number of steps (default: 50)
        - input.guidance_scale (optional): Guidance scale (default: 7.5)
        - input.strength (optional): Strength for img2img (0.0-1.0, default: 0.75)
        - input.width (optional): Image width (default: 512)
        - input.height (optional): Image height (default: 512)
       
    Returns:
        dict: Contains the generated image as base64 string
    """
    
    global pipe, img2img_pipe
    
    try:
        # Load model if not already loaded
        if pipe is None:
            pipe, img2img_pipe = load_model()
        
        # Extract input data (RunPod uses 'input' key in job dict)
        print("Worker Start")
        input_data = job.get('input', {})
        
        prompt = input_data.get('prompt', 'kawaii pastel style, cute, soft pastel colors, anime style, adorable')
        negative_prompt = input_data.get('negative_prompt', 'blurry, low quality, distorted, nsfw')
        num_inference_steps = input_data.get('num_inference_steps', 50)
        guidance_scale = input_data.get('guidance_scale', 7.5)
        strength = input_data.get('strength', 0.75)
        
        # Check if img2img mode (has input image)
        input_image_b64 = input_data.get('image', None)
        
        if input_image_b64:
            # IMG2IMG MODE
            print("=" * 50)
            print("IMG2IMG MODE: Converting image to kawaii pastel style")
            print("=" * 50)
            
            # Decode base64 image
            try:
                # Remove data URL prefix if present
                if ',' in input_image_b64:
                    input_image_b64 = input_image_b64.split(',')[1]
                
                image_bytes = base64.b64decode(input_image_b64)
                input_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                
                # Resize if needed (maintain aspect ratio, max 1024px)
                max_size = 1024
                if max(input_image.size) > max_size:
                    ratio = max_size / max(input_image.size)
                    new_size = (int(input_image.size[0] * ratio), int(input_image.size[1] * ratio))
                    input_image = input_image.resize(new_size, Image.Resampling.LANCZOS)
                
                print(f"Input image size: {input_image.size}")
                print(f"Steps: {num_inference_steps}")
                print(f"Strength: {strength} (higher = more transformation)")
                print(f"Prompt: {prompt}")
                print("Converting image...")
                
            except Exception as decode_error:
                print(f"Error decoding input image: {decode_error}")
                return {
                    "error": f"Failed to decode input image: {str(decode_error)}",
                    "status": "error"
                }
            
            # Generate image using img2img
            result = img2img_pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=input_image,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                strength=strength
            )
            image = result.images[0]
            
        else:
            # TEXT2IMG MODE
            print("=" * 50)
            print("TEXT2IMG MODE: Generating image from text")
            print("=" * 50)
            
            width = input_data.get('width', 512)
            height = input_data.get('height', 512)
            
            print(f"Received prompt: {prompt}")
            print(f"Generating image with {num_inference_steps} steps...")
            print(f"Size: {width}x{height}")
            
            # Generate image
            image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height
            ).images[0]
        
        # Convert image to base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        print("✓ Image converted successfully!")
        
        return {
            "image": img_str,
            "prompt": prompt,
            "status": "success"
        }
        
    except Exception as e:
        print(f"Error in handler: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "error": str(e),
            "status": "error"
        }

# Start the Serverless function when the script is run
if __name__ == '__main__':
    try:
        print("=" * 50)
        print("Starting Kawaii Pastel Worker...")
        print("=" * 50)
        
        # Print system info
        import sys
        print(f"Python version: {sys.version}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print("=" * 50)
        
        # Pre-load the model
        print("Pre-loading model...")
        print("This may take several minutes on first run...")
        pipe, img2img_pipe = load_model()
        print("=" * 50)
        print("Model loaded successfully! Starting serverless worker...")
        print("=" * 50)
        
        runpod.serverless.start({'handler': handler})
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
        sys.exit(0)
    except Exception as e:
        print("=" * 50)
        print("FATAL ERROR during startup:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print("=" * 50)
        import traceback
        print("Full traceback:")
        traceback.print_exc()
        print("=" * 50)
        # Force flush output before exit
        import sys
        sys.stdout.flush()
        sys.stderr.flush()
        # Re-raise to exit with error code
        raise
