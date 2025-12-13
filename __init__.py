"""
ComfyUI Text-to-Pose Custom Nodes
================================

Generate human poses from text descriptions using the T2P Transformer model.

Installation:
1. Clone this repository into ComfyUI/custom_nodes/
2. Run: python install.py
3. Install requirements: pip install -r requirements.txt
4. Restart ComfyUI

Usage:
1. Add "T2P Model Loader" node to load the model
2. Add "Text to Pose" node and connect the model
3. Enter your text prompt describing the pose
4. Connect output to ControlNet or T2I-Adapter nodes
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

# Version info
__version__ = "1.0.0"
__author__ = "Based on clement-bonnet/text-to-pose"
