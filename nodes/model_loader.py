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
                "dtype": (["auto", "float32", "float16", "bfloat16"], {"default": "auto"}),
            }
        }
    
    RETURN_TYPES = ("T2P_MODEL",)
    RETURN_NAMES = ("t2p_model",)
    FUNCTION = "load_model"
    CATEGORY = "text-to-pose"
    
    def load_model(self, model_name, device="auto", dtype="auto"):
        from t2p.model import T2PTransformer
        
        # Determine device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device)
        
        # Determine dtype - default to float32 for stability (especially on ROCm)
        if dtype == "auto":
            torch_dtype = torch.float32  # Use float32 by default for stability
        else:
            dtype_map = {
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }
            torch_dtype = dtype_map[dtype]
        
        # Load model from HuggingFace
        model_id = self.MODELS[model_name]
        print(f"[T2P] Loading model from {model_id}...")
        
        t2p_model = T2PTransformer.from_pretrained(model_id)
        t2p_model.to(device=device, dtype=torch_dtype)
        t2p_model.eval()
        
        print(f"[T2P] Model loaded successfully on {device} with dtype {torch_dtype}")
        
        return ({"model": t2p_model, "device": device, "dtype": torch_dtype},)


NODE_CLASS_MAPPINGS = {
    "T2PModelLoader": T2PModelLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "T2PModelLoader": "T2P Model Loader",
}
