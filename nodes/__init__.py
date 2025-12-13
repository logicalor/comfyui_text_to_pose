"""
ComfyUI Text-to-Pose Custom Nodes
================================

This package provides nodes for generating human poses from text descriptions
using the Text-to-Pose (T2P) Transformer model.

Nodes included:
- T2PModelLoader: Load the T2P transformer model
- TextToPose: Generate pose from text prompt
- TextToPoseBatch: Generate multiple pose variations
- T2IPoseAdapterLoader: Load T2I-Adapter for SDXL
- ApplyT2IPoseAdapter: Apply pose conditioning to SDXL

Based on the paper "From Text to Pose to Image: Improving Diffusion Model Control and Quality"
https://github.com/clement-bonnet/text-to-pose
"""

from .model_loader import NODE_CLASS_MAPPINGS as MODEL_LOADER_NODES
from .model_loader import NODE_DISPLAY_NAME_MAPPINGS as MODEL_LOADER_NAMES
from .text_to_pose import NODE_CLASS_MAPPINGS as TEXT_TO_POSE_NODES
from .text_to_pose import NODE_DISPLAY_NAME_MAPPINGS as TEXT_TO_POSE_NAMES
from .adapter_loader import NODE_CLASS_MAPPINGS as ADAPTER_NODES
from .adapter_loader import NODE_DISPLAY_NAME_MAPPINGS as ADAPTER_NAMES

# Combine all node mappings
NODE_CLASS_MAPPINGS = {
    **MODEL_LOADER_NODES,
    **TEXT_TO_POSE_NODES,
    **ADAPTER_NODES,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **MODEL_LOADER_NAMES,
    **TEXT_TO_POSE_NAMES,
    **ADAPTER_NAMES,
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
