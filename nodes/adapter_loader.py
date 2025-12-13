"""
T2I-Adapter Loader Node for ComfyUI
Loads the DWPose T2I-Adapter for SDXL from HuggingFace
"""

import torch


class T2IPoseAdapterLoader:
    """
    Loads the T2I-Adapter trained on DWPose for SDXL.
    This adapter can be used with SDXL to generate images conditioned on pose.
    """
    
    ADAPTERS = {
        "t2i-adapter-sdxl-dwpose": "clement-bonnet/t2i-adapter-sdxl-dwpose",
    }
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "adapter_name": (list(cls.ADAPTERS.keys()), {"default": "t2i-adapter-sdxl-dwpose"}),
            },
            "optional": {
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
                "dtype": (["auto", "float16", "float32", "bfloat16"], {"default": "auto"}),
            }
        }
    
    RETURN_TYPES = ("T2I_ADAPTER",)
    RETURN_NAMES = ("t2i_adapter",)
    FUNCTION = "load_adapter"
    CATEGORY = "text-to-pose"
    
    def load_adapter(self, adapter_name, device="auto", dtype="auto"):
        from diffusers import T2IAdapter
        
        # Determine device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device)
        
        # Determine dtype
        if dtype == "auto":
            if device.type == "cuda":
                torch_dtype = torch.float16
            else:
                torch_dtype = torch.float32
        else:
            dtype_map = {
                "float16": torch.float16,
                "float32": torch.float32,
                "bfloat16": torch.bfloat16,
            }
            torch_dtype = dtype_map[dtype]
        
        # Load adapter from HuggingFace
        adapter_id = self.ADAPTERS[adapter_name]
        print(f"[T2P] Loading T2I-Adapter from {adapter_id}...")
        
        adapter = T2IAdapter.from_pretrained(
            adapter_id,
            torch_dtype=torch_dtype,
        )
        adapter.to(device)
        
        print(f"[T2P] T2I-Adapter loaded successfully on {device} with dtype {torch_dtype}")
        
        return ({
            "adapter": adapter,
            "device": device,
            "dtype": torch_dtype,
        },)


class ApplyT2IPoseAdapter:
    """
    Applies the T2I-Adapter to condition SDXL generation on a pose image.
    This node integrates with the standard ComfyUI SDXL workflow.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "t2i_adapter": ("T2I_ADAPTER",),
                "pose_image": ("IMAGE",),
                "adapter_conditioning_scale": ("FLOAT", {
                    "default": 0.8,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.05,
                    "tooltip": "Strength of pose conditioning (0 = no effect, 1 = full effect)"
                }),
            },
        }
    
    RETURN_TYPES = ("T2I_ADAPTER_CONDITIONING",)
    RETURN_NAMES = ("adapter_conditioning",)
    FUNCTION = "apply_adapter"
    CATEGORY = "text-to-pose"
    
    def apply_adapter(self, t2i_adapter, pose_image, adapter_conditioning_scale):
        """
        Prepares adapter conditioning from a pose image.
        The output can be used with SDXL pipelines that support T2I-Adapter.
        """
        adapter = t2i_adapter["adapter"]
        device = t2i_adapter["device"]
        dtype = t2i_adapter["dtype"]
        
        # Convert ComfyUI image (B, H, W, C) to diffusers format (B, C, H, W)
        # Also convert from [0, 1] float to proper format
        pose_tensor = pose_image.permute(0, 3, 1, 2).to(device=device, dtype=dtype)
        
        return ({
            "adapter": adapter,
            "adapter_image": pose_tensor,
            "adapter_conditioning_scale": adapter_conditioning_scale,
        },)


class T2PFullPipeline:
    """
    Complete Text-to-Pose-to-Image pipeline using SDXL with T2I-Adapter.
    Combines pose generation and image generation in one node.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "t2p_model": ("T2P_MODEL",),
                "t2i_adapter": ("T2I_ADAPTER",),
                "sdxl_model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "pose_prompt": ("STRING", {
                    "multiline": True,
                    "default": "a person standing with arms crossed"
                }),
                "image_prompt": ("STRING", {
                    "multiline": True,
                    "default": "professional photograph of a person, high quality, detailed"
                }),
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": "blurry, low quality, deformed, ugly"
                }),
                "width": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 64}),
                "height": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 64}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 30, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 7.5, "min": 1.0, "max": 20.0, "step": 0.5}),
            },
            "optional": {
                "adapter_scale": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.05}),
                "pose_temperature": ("FLOAT", {"default": 0.1, "min": 0.01, "max": 2.0, "step": 0.01}),
                "num_poses": ("INT", {"default": 1, "min": 1, "max": 5}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE",)
    RETURN_NAMES = ("generated_image", "pose_image",)
    FUNCTION = "generate"
    CATEGORY = "text-to-pose"
    
    def generate(self, t2p_model, t2i_adapter, sdxl_model, clip, vae,
                 pose_prompt, image_prompt, negative_prompt,
                 width, height, seed, steps, cfg,
                 adapter_scale=0.8, pose_temperature=0.1, num_poses=1):
        """
        Full pipeline: Text → Pose → Image
        
        Note: This is a simplified implementation. For full functionality,
        users should use separate nodes connected in a workflow.
        """
        # This is a placeholder for the full pipeline
        # In practice, users will connect individual nodes
        raise NotImplementedError(
            "Full pipeline node is not yet implemented. "
            "Please use individual nodes (T2PModelLoader → TextToPose → "
            "T2IPoseAdapterLoader → ApplyT2IPoseAdapter) connected to your SDXL workflow."
        )


NODE_CLASS_MAPPINGS = {
    "T2IPoseAdapterLoader": T2IPoseAdapterLoader,
    "ApplyT2IPoseAdapter": ApplyT2IPoseAdapter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "T2IPoseAdapterLoader": "T2I Pose Adapter Loader",
    "ApplyT2IPoseAdapter": "Apply T2I Pose Adapter",
}
