from diffusers import StableDiffusionPipeline
import torch

class CFG:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42
    generator = torch.Generator(device).manual_seed(seed)
    image_gen_steps = 1
    image_gen_model_id = "stabilityai/stable-diffusion-2"
    image_gen_size = (400, 400)
    image_gen_guidance_scale = 9

def load_model(model_id, device):
    """
    Load the Stable Diffusion model.
    
    Args:
        model_id (str): The ID of the pre-trained model.
        device (str): The device to load the model onto ("cuda" or "cpu").
    
    Returns:
        model: The loaded Stable Diffusion model.
    """
    # Load the model without specifying precision
    pipe = StableDiffusionPipeline.from_pretrained(model_id)
    pipe.to(device)
    return pipe

def generate_image(prompt, model, cfg):
    """
    Generate an image from a text prompt using a pre-trained model.
    
    Args:
        prompt (str): The text prompt for image generation.
        model: The pre-trained image generation model.
        cfg: Configuration object with generation parameters.
    
    Returns:
        PIL.Image.Image: The generated image, resized to the specified dimensions.
    """
    # Validate inputs
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Prompt must be a non-empty string.")
    
    if not hasattr(model, 'to') or not callable(getattr(model, 'to')):
        raise TypeError("The provided model is not compatible or lacks the 'to' method.")
    
    # Move model to the specified device
    model.to(cfg.device)

    try:
        # Generate the image
        with torch.no_grad():
            result = model(
                prompt,
                num_inference_steps=cfg.image_gen_steps,
                generator=cfg.generator,
                guidance_scale=cfg.image_gen_guidance_scale
            )
        
        # Ensure the result has images
        if not hasattr(result, 'images') or not result.images:
            raise RuntimeError("The model did not return any images.")

        # Extract and process the image
        image = result.images[0]
        image = image.convert("RGB")  # Ensure the image is in RGB mode
        image = image.resize(cfg.image_gen_size, Image.LANCZOS)  # High-quality resizing
        
        # Display the image in a notebook
        display(image)
        
        return image
    
    except RuntimeError as e:
        print(f"Runtime error during image generation: {e}")
        raise
    except Exception as e:
        print(f"An error occurred during image generation: {e}")
        raise

# Load the model
model = load_model(CFG.image_gen_model_id, CFG.device)

# Generate and display the image
try:
    image = generate_image("astronaut in space", model, CFG)
    # Optionally save the image
    # image.save('generated_image.png')
except Exception as e:
    print(f"Failed to generate image: {e}")
