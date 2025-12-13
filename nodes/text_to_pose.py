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
    The number of people in the image is automatically determined from the prompt.
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
    
    def generate_pose(self, t2p_model, prompt, width, height, seed,
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
        
        # Disable autocast to avoid mixed precision issues on ROCm
        with torch.no_grad(), torch.amp.autocast(device_type='cuda', enabled=False):
            # Ensure model is in the correct dtype
            model = model.to(dtype=dtype)
            
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
            
            # Check for NaN in embeddings
            if torch.isnan(text_embeddings).any():
                print("[T2P] Warning: NaN detected in text embeddings, using CPU fallback")
                # Fallback to CPU
                model = model.to("cpu")
                input_ids = input_ids.to("cpu")
                text_outputs = model.clip_text_model(
                    input_ids,
                    output_hidden_states=True
                )
                text_embeddings = text_outputs.hidden_states[-2].squeeze(0).float()
            
            # Generate pose samples
            # The model automatically determines the number of people from the prompt
            # (supports up to 5 people per image)
            poses = model.generate(
                text_embeddings=text_embeddings,
                num_poses=1,  # Batch size of 1 (single image output)
                bbox_dist_temperature=bbox_temperature,
                pose_dist_temperature=pose_temperature,
                image_ratio=image_ratio,
            )
            
            # Convert to DWPose format
            dw_pose = model.convert_to_dwpose(poses)
            
            # Handle case where convert_to_dwpose returns a list (batch output)
            if isinstance(dw_pose, list):
                dw_pose = dw_pose[0]  # Take the first result
            
            # Debug: Print pose info
            bodies = dw_pose.get("bodies", {})
            candidate = bodies.get("candidate")
            if candidate is not None:
                candidate = np.array(candidate) if not isinstance(candidate, np.ndarray) else candidate
                print(f"[T2P] Pose candidate shape: {candidate.shape}")
                print(f"[T2P] Pose candidate min/max: {candidate.min():.2f} / {candidate.max():.2f}")
                
                # Scale coordinates if they appear to be normalized (0-1 range)
                if candidate.max() <= 1.0:
                    print(f"[T2P] Scaling normalized coordinates to {width}x{height}")
                    candidate[:, 0] *= width
                    candidate[:, 1] *= height
                    dw_pose["bodies"]["candidate"] = candidate
            
            # Scale faces if present and normalized
            faces = dw_pose.get("faces")
            if faces is not None:
                faces = np.array(faces) if not isinstance(faces, np.ndarray) else faces
                if faces.size > 0 and faces.max() <= 1.0:
                    print(f"[T2P] Scaling face coordinates to {width}x{height}")
                    faces[..., 0] *= width
                    faces[..., 1] *= height
                dw_pose["faces"] = faces
            
            # Scale hands if present and normalized
            hands = dw_pose.get("hands")
            if hands is not None:
                hands = np.array(hands) if not isinstance(hands, np.ndarray) else hands
                if hands.size > 0 and hands.max() <= 1.0:
                    print(f"[T2P] Scaling hand coordinates to {width}x{height}")
                    hands[..., 0] *= width
                    hands[..., 1] *= height
                dw_pose["hands"] = hands
        
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
    The number of people per image is automatically determined from the prompt.
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
    
    def generate_batch(self, t2p_model, prompt, width, height,
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
                    num_poses=1,  # Batch size of 1 per iteration
                    bbox_dist_temperature=bbox_temperature,
                    pose_dist_temperature=pose_temperature,
                    image_ratio=image_ratio,
                )
                
                dw_pose = model.convert_to_dwpose(poses)
                
                # Handle case where convert_to_dwpose returns a list
                if isinstance(dw_pose, list):
                    dw_pose = dw_pose[0]
                
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
