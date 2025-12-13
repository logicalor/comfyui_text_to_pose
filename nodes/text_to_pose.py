"""
Text to Pose Generator Node for ComfyUI
Generates pose images and keypoints from text prompts
"""

import torch
import numpy as np
from PIL import Image


class TextToPose:
    """
    Generates human poses from text descriptions using the T2P Transformer.
    Outputs both a rendered pose image and raw keypoints data.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "t2p_model": ("T2P_MODEL",),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "a person standing with arms raised"
                }),
                "width": ("INT", {
                    "default": 1024,
                    "min": 256,
                    "max": 2048,
                    "step": 64
                }),
                "height": ("INT", {
                    "default": 1024,
                    "min": 256,
                    "max": 2048,
                    "step": 64
                }),
                "num_poses": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 5,
                    "step": 1,
                    "tooltip": "Number of people/poses to generate"
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff
                }),
            },
            "optional": {
                "bbox_temperature": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.01,
                    "max": 2.0,
                    "step": 0.01,
                    "tooltip": "Temperature for bounding box sampling (lower = more deterministic)"
                }),
                "pose_temperature": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.01,
                    "max": 2.0,
                    "step": 0.01,
                    "tooltip": "Temperature for pose sampling (lower = more deterministic)"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "POSE_KEYPOINTS",)
    RETURN_NAMES = ("pose_image", "pose_keypoints",)
    FUNCTION = "generate_pose"
    CATEGORY = "text-to-pose"
    
    def generate_pose(self, t2p_model, prompt, width, height, num_poses, seed,
                      bbox_temperature=0.1, pose_temperature=0.1):
        from transformers import AutoProcessor
        from .pose_utils import draw_pose
        
        model = t2p_model["model"]
        device = t2p_model["device"]
        dtype = t2p_model.get("dtype", torch.float32)
        
        # Set seed for reproducibility
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        
        # Calculate image ratio (width / height)
        image_ratio = width / height
        
        # Get CLIP processor for tokenization
        clip_processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
        
        with torch.no_grad():
            # Tokenize prompt
            inputs = clip_processor(
                text=prompt,
                return_tensors="pt",
                padding="max_length",
                truncation=True
            )
            input_ids = inputs["input_ids"].to(device)
            
            # Get text embeddings from CLIP (using second-to-last hidden state)
            text_outputs = model.clip_text_model(
                input_ids,
                output_hidden_states=True
            )
            text_embeddings = text_outputs.hidden_states[-2].squeeze(0).to(dtype)
            
            # Generate pose samples
            poses = model.generate(
                text_embeddings=text_embeddings,
                num_poses=num_poses,
                bbox_dist_temperature=bbox_temperature,
                pose_dist_temperature=pose_temperature,
                image_ratio=image_ratio,
            )
            
            # Convert to DWPose format
            dw_pose = model.convert_to_dwpose(poses)
        
        # Render pose image
        pose_image_np = draw_pose(dw_pose, height, width)
        
        # Convert to ComfyUI IMAGE format (B, H, W, C) float32 [0, 1]
        pose_image = torch.from_numpy(pose_image_np).float() / 255.0
        pose_image = pose_image.unsqueeze(0)  # Add batch dimension
        
        # Prepare keypoints output (convert numpy arrays to lists for serialization)
        pose_keypoints = {
            "bodies": {
                "candidate": dw_pose["bodies"]["candidate"].tolist() if isinstance(dw_pose["bodies"]["candidate"], np.ndarray) else dw_pose["bodies"]["candidate"],
                "subset": dw_pose["bodies"]["subset"].tolist() if isinstance(dw_pose["bodies"]["subset"], np.ndarray) else dw_pose["bodies"]["subset"],
            },
            "faces": dw_pose["faces"].tolist() if isinstance(dw_pose["faces"], np.ndarray) else dw_pose["faces"],
            "hands": dw_pose["hands"].tolist() if isinstance(dw_pose["hands"], np.ndarray) else dw_pose["hands"],
            "width": width,
            "height": height,
        }
        
        return (pose_image, pose_keypoints,)


class TextToPoseBatch:
    """
    Generates multiple pose variations from a single text prompt.
    Useful for exploring different pose interpretations.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "t2p_model": ("T2P_MODEL",),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "a person dancing"
                }),
                "width": ("INT", {
                    "default": 1024,
                    "min": 256,
                    "max": 2048,
                    "step": 64
                }),
                "height": ("INT", {
                    "default": 1024,
                    "min": 256,
                    "max": 2048,
                    "step": 64
                }),
                "num_poses_per_image": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 5,
                    "step": 1,
                    "tooltip": "Number of people per image"
                }),
                "batch_size": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 16,
                    "step": 1,
                    "tooltip": "Number of different pose variations to generate"
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff
                }),
            },
            "optional": {
                "bbox_temperature": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.01,
                    "max": 2.0,
                    "step": 0.01,
                }),
                "pose_temperature": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.01,
                    "max": 2.0,
                    "step": 0.01,
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("pose_images",)
    FUNCTION = "generate_batch"
    CATEGORY = "text-to-pose"
    
    def generate_batch(self, t2p_model, prompt, width, height, num_poses_per_image,
                       batch_size, seed, bbox_temperature=0.3, pose_temperature=0.3):
        from transformers import AutoProcessor
        from .pose_utils import draw_pose
        
        model = t2p_model["model"]
        device = t2p_model["device"]
        dtype = t2p_model.get("dtype", torch.float32)
        
        image_ratio = width / height
        clip_processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
        
        pose_images = []
        
        with torch.no_grad():
            # Tokenize prompt once
            inputs = clip_processor(
                text=prompt,
                return_tensors="pt",
                padding="max_length",
                truncation=True
            )
            input_ids = inputs["input_ids"].to(device)
            
            text_outputs = model.clip_text_model(
                input_ids,
                output_hidden_states=True
            )
            text_embeddings = text_outputs.hidden_states[-2].squeeze(0).to(dtype)
            
            # Generate batch_size different poses
            for i in range(batch_size):
                # Set seed for this iteration
                current_seed = seed + i
                torch.manual_seed(current_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(current_seed)
                
                poses = model.generate(
                    text_embeddings=text_embeddings,
                    num_poses=num_poses_per_image,
                    bbox_dist_temperature=bbox_temperature,
                    pose_dist_temperature=pose_temperature,
                    image_ratio=image_ratio,
                )
                
                dw_pose = model.convert_to_dwpose(poses)
                pose_image_np = draw_pose(dw_pose, height, width)
                
                pose_tensor = torch.from_numpy(pose_image_np).float() / 255.0
                pose_images.append(pose_tensor)
        
        # Stack into batch (B, H, W, C)
        batch_tensor = torch.stack(pose_images, dim=0)
        
        return (batch_tensor,)


NODE_CLASS_MAPPINGS = {
    "TextToPose": TextToPose,
    "TextToPoseBatch": TextToPoseBatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TextToPose": "Text to Pose",
    "TextToPoseBatch": "Text to Pose (Batch)",
}
