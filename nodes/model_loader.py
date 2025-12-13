"""
T2P Model Loader Node for ComfyUI
Loads the Text-to-Pose transformer model from HuggingFace
"""

import os
import sys

# Add parent directory to path so t2p module can be found
_current_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

import torch


def get_comfyui_models_dir():
    """Get ComfyUI's models directory, with fallback options."""
    # Try to import from ComfyUI's folder_paths
    try:
        import folder_paths
        # Use ComfyUI's models directory
        models_dir = folder_paths.models_dir
        t2p_dir = os.path.join(models_dir, "t2p")
        os.makedirs(t2p_dir, exist_ok=True)
        return t2p_dir
    except ImportError:
        pass
    
    # Fallback: try to find ComfyUI models dir relative to this file
    # This file is in custom_nodes/text-to-pose/nodes/
    custom_nodes_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    comfyui_dir = os.path.dirname(custom_nodes_dir)
    models_dir = os.path.join(comfyui_dir, "models", "t2p")
    if os.path.exists(os.path.dirname(models_dir)):
        os.makedirs(models_dir, exist_ok=True)
        return models_dir
    
    # Final fallback: use user's cache directory
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "t2p")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def get_local_model_path(model_name: str) -> str:
    """Get the local path where a model should be stored."""
    models_dir = get_comfyui_models_dir()
    return os.path.join(models_dir, model_name)


def is_model_downloaded(model_name: str) -> bool:
    """Check if a model has already been downloaded locally."""
    model_path = get_local_model_path(model_name)
    # Check for the model config file which indicates a complete download
    config_file = os.path.join(model_path, "config.json")
    model_file = os.path.join(model_path, "model.safetensors")
    # Also check for pytorch model file as fallback
    pytorch_file = os.path.join(model_path, "pytorch_model.bin")
    return os.path.exists(config_file) and (os.path.exists(model_file) or os.path.exists(pytorch_file))


class T2PModelLoader:
    """
    Loads the Text-to-Pose (T2P) Transformer model from HuggingFace.
    The model converts text prompts into pose keypoints.
    """
    
    MODELS = {
        "t2p-transformer-v0": "clement-bonnet/t2p-transformer-v0",
    }
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (list(cls.MODELS.keys()), {"default": "t2p-transformer-v0"}),
            },
            "optional": {
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
                "force_cpu": ("BOOLEAN", {"default": False, "tooltip": "Force CPU mode (slower but more compatible)"}),
            }
        }
    
    RETURN_TYPES = ("T2P_MODEL",)
    RETURN_NAMES = ("t2p_model",)
    FUNCTION = "load_model"
    CATEGORY = "text-to-pose"
    
    def load_model(self, model_name, device="auto", force_cpu=False):
        from t2p.model import T2PTransformer
        
        # Determine device
        if force_cpu:
            device = "cpu"
        elif device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device)
        
        # Always use float32 - MultivariateNormal distribution doesn't support half precision
        torch_dtype = torch.float32
        
        # Get model repo ID
        model_id = self.MODELS[model_name]
        local_path = get_local_model_path(model_name)
        
        # Check if model exists locally
        if is_model_downloaded(model_name):
            print(f"[T2P] Loading model from local cache: {local_path}")
            t2p_model = T2PTransformer.from_pretrained(local_path)
        else:
            print(f"[T2P] Downloading model from {model_id} to {local_path}...")
            t2p_model = T2PTransformer.from_pretrained(model_id)
            # Save to local directory for future use
            print(f"[T2P] Saving model to local cache: {local_path}")
            os.makedirs(local_path, exist_ok=True)
            t2p_model.save_pretrained(local_path)
        
        t2p_model.to(device=device, dtype=torch_dtype)
        t2p_model.eval()
        
        print(f"[T2P] Model loaded successfully on {device}")
        
        return ({"model": t2p_model, "device": device, "dtype": torch_dtype},)


NODE_CLASS_MAPPINGS = {
    "T2PModelLoader": T2PModelLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "T2PModelLoader": "T2P Model Loader",
}
