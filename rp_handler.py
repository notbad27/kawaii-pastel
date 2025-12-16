import runpod
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from safetensors.torch import load_file
import base64
import io
from PIL import Image
import os

# Global variables to store the loaded models
pipe = None  # Text2Img pipeline
img2img_pipe = None  # Img2Img pipeline

# Configure HuggingFace cache location
# Priority: Pre-built cache > Network volume > Container disk (/tmp with more space)
print("=" * 50)
print("Checking available storage locations...")
print("=" * 50)

# Check for pre-built model cache from Docker build
DEFAULT_CACHE = '/root/.cache/huggingface'
if os.path.exists(DEFAULT_CACHE):
    # Check if it actually has the model
    model_exists = os.path.exists(os.path.join(DEFAULT_CACHE, 'hub')) and \
                   len(os.listdir(os.path.join(DEFAULT_CACHE, 'hub'))) > 0
    if model_exists:
        hf_cache_dir = DEFAULT_CACHE
        os.environ['HF_HOME'] = hf_cache_dir
        os.environ['HUGGINGFACE_HUB_CACHE'] = os.path.join(hf_cache_dir, 'hub')
        print(f"✓ Using pre-built model cache: {hf_cache_dir}")
        NETWORK_VOLUME_PATH = None
    else:
        NETWORK_VOLUME_PATH = None
        # Check network volume mount points
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
else:
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
            
            # Try loading from pre-built cache (should always work if Docker build succeeded)
            load_kwargs = {
                'torch_dtype': torch.float16 if torch.cuda.is_available() else torch.float32,
            }
            if cache_dir:
                load_kwargs['cache_dir'] = cache_dir

            pipe = StableDiffusionPipeline.from_pretrained(
                base_model,
                **load_kwargs
            )
            print("✓ Loaded base model from cache!")
            
            # Load custom weights from safetensors file
            print(f"Loading custom weights from: {model_path}")
            try:
                state_dict = load_file(model_path)
                print(f"Loaded {len(state_dict)} weight tensors from safetensors file")
                
                # Apply custom weights to UNet (most safetensors files contain UNet weights)
                # Try UNet first, then VAE if UNet fails
                try:
                    missing_keys, unexpected_keys = pipe.unet.load_state_dict(state_dict, strict=False)
                    print(f"Applied weights to UNet - Missing: {len(missing_keys)}, Unexpected: {len(unexpected_keys)}")
                    print("✓ Custom UNet weights applied successfully!")
                except Exception as unet_error:
                    print(f"Could not apply to UNet: {unet_error}")
                    # Try VAE as fallback
                    try:
                        missing_keys, unexpected_keys = pipe.vae.load_state_dict(state_dict, strict=False)
                        print(f"Applied weights to VAE - Missing: {len(missing_keys)}, Unexpected: {len(unexpected_keys)}")
                        print("✓ Custom VAE weights applied successfully!")
                    except Exception as vae_error:
                        print(f"Could not apply to VAE either: {vae_error}")
                        print("Warning: Custom weights not applied. Using base model only.")
                
            except Exception as weight_error:
                print(f"Warning: Could not load custom weights: {weight_error}")
                print("Using base model without custom weights.")
            
            if torch.cuda.is_available():
                pipe = pipe.to("cuda")
            
            try:
                pipe.enable_xformers_memory_efficient_attention()
            except:
                print("xformers not available, using default attention")
            
            # Create img2img pipeline from the same model
            print("Creating img2img pipeline...")
            img2img_pipe = StableDiffusionImg2ImgPipeline(
                vae=pipe.vae,
                text_encoder=pipe.text_encoder,
                tokenizer=pipe.tokenizer,
                unet=pipe.unet,
                scheduler=pipe.scheduler,
                safety_checker=pipe.safety_checker,
                feature_extractor=pipe.feature_extractor,
            )
            
            if torch.cuda.is_available():
                img2img_pipe = img2img_pipe.to("cuda")
            
            try:
                img2img_pipe.enable_xformers_memory_efficient_attention()
            except:
                pass
            
            print("✓ Model loaded successfully!")
            print("✓ Img2Img pipeline ready!")
            
        except Exception as e2:
                print(f"Error loading model: {e2}")
                import traceback
                traceback.print_exc()
                raise e2
    
    return pipe, img2img_pipe

def handler(job):
    """
    This function processes incoming requests to convert images to kawaii pastel style.
    
    Supports both:
    - Text2Img: Generate from text prompt
    - Img2Img: Convert real image to kawaii pastel style
    
    Args:
        job (dict): Contains the input data and request metadata
        - input.prompt: Text prompt (required)
        - input.image (optional): Base64 encoded input image for img2img conversion
        - input.negative_prompt (optional): Negative prompt
        - input.num_inference_steps (optional): Number of steps (default: 50)
        - input.guidance_scale (optional): Guidance scale (default: 7.5)
        - input.strength (optional): Img2Img strength 0.0-1.0 (default: 0.75)
        - input.width (optional): Image width (default: 512, only for text2img)
        - input.height (optional): Image height (default: 512, only for text2img)
       
    Returns:
        dict: Contains the generated/converted image as base64 string
    """
    
    global pipe, img2img_pipe
    
    try:
        # Load model if not already loaded
        if pipe is None or img2img_pipe is None:
            pipe, img2img_pipe = load_model()
        
        # Extract input data (RunPod uses 'input' key in job dict)
        print("Worker Start")
        input_data = job.get('input', {})
        
        prompt = input_data.get('prompt', 'a cute kawaii pastel character')
        negative_prompt = input_data.get('negative_prompt', 'blurry, low quality, distorted')
        num_inference_steps = input_data.get('num_inference_steps', 50)
        guidance_scale = input_data.get('guidance_scale', 7.5)
        strength = input_data.get('strength', 0.75)  # For img2img
        
        # Check if input image is provided (img2img mode)
        input_image_b64 = input_data.get('image', None)
        
        if input_image_b64:
            # IMG2IMG MODE: Convert real image to kawaii pastel
            print("=" * 50)
            print("IMG2IMG MODE: Converting image to kawaii pastel style")
            print("=" * 50)
            
            # Decode base64 image
            try:
                image_data = base64.b64decode(input_image_b64)
                init_image = Image.open(io.BytesIO(image_data)).convert("RGB")
                print(f"Input image size: {init_image.size}")
            except Exception as img_error:
                return {
                    "error": f"Failed to decode input image: {str(img_error)}",
                    "status": "error"
                }
            
            print(f"Prompt: {prompt}")
            print(f"Strength: {strength} (higher = more transformation)")
            print(f"Steps: {num_inference_steps}")
            print("Converting image...")
            
            # Convert image using img2img pipeline
            image = img2img_pipe(
                prompt=prompt,
                image=init_image,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                strength=strength  # How much to transform (0.0 = original, 1.0 = completely new)
            ).images[0]
            
            print("✓ Image converted successfully!")
            
        else:
            # TEXT2IMG MODE: Generate from text prompt
            print("=" * 50)
            print("TEXT2IMG MODE: Generating image from text")
            print("=" * 50)
            
            width = input_data.get('width', 512)
            height = input_data.get('height', 512)
            
            print(f"Prompt: {prompt}")
            print(f"Size: {width}x{height}")
            print(f"Steps: {num_inference_steps}")
            print("Generating image...")
            
            # Generate image from text
            image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height
            ).images[0]
            
            print("✓ Image generated successfully!")
        
        # Convert image to base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return {
            "image": img_str,
            "prompt": prompt,
            "mode": "img2img" if input_image_b64 else "text2img",
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
        
        # Check if model file exists
        model_path = "/nyl_kawaii_pastel.safetensors"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        print(f"✓ Model file found: {model_path}")
        
        # Check network volume
        print(f"Checking network volume at /workspace...")
        if os.path.exists('/workspace'):
            print(f"✓ /workspace exists")
            if os.path.exists('/workspace/.cache/huggingface/hub/models--runwayml--stable-diffusion-v1-5'):
                print(f"✓ Model cache found in network volume!")
            else:
                print(f"⚠ Model cache NOT found in network volume")
                print(f"  Expected: /workspace/.cache/huggingface/hub/models--runwayml--stable-diffusion-v1-5")
        else:
            print(f"⚠ /workspace does not exist - network volume may not be mounted")
        
        # Pre-load the model
        print("Pre-loading model...")
        print("This may take several minutes on first run...")
        load_model()  # This loads both text2img and img2img pipelines
        print("=" * 50)
        print("Model loaded successfully! Starting serverless worker...")
        print("=" * 50)
        
        runpod.serverless.start({'handler': handler})
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
        import sys
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
