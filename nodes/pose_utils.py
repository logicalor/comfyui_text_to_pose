"""
Pose rendering utilities for ComfyUI Text-to-Pose nodes
Handles drawing DWPose format poses to images
"""

import numpy as np
from PIL import Image

# Color definitions for pose visualization
BODY_COLORS = [
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0),
    (170, 255, 0), (85, 255, 0), (0, 255, 0), (0, 255, 85),
    (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
    (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255),
    (255, 0, 170), (255, 0, 85),
]

LIMB_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # Right arm
    (1, 5), (5, 6), (6, 7),          # Left arm
    (1, 8), (8, 9), (9, 10),         # Right leg
    (1, 11), (11, 12), (12, 13),     # Left leg
    (0, 14), (14, 16),               # Right eye/ear
    (0, 15), (15, 17),               # Left eye/ear
]


def draw_pose(pose_dict, height, width):
    """
    Draw a complete pose (body, face, hands) from DWPose format.
    
    Args:
        pose_dict: Dictionary with 'bodies', 'faces', 'hands' keys
        height: Output image height
        width: Output image width
    
    Returns:
        numpy array (H, W, 3) RGB image
    """
    try:
        # Try to use controlnet_aux's drawing functions first
        from controlnet_aux.dwpose.util import draw_bodypose, draw_facepose, draw_handpose
        
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Draw body
        if pose_dict.get("bodies") is not None:
            bodies = pose_dict["bodies"]
            candidate = bodies.get("candidate")
            subset = bodies.get("subset")
            if candidate is not None and subset is not None:
                candidate = np.array(candidate) if not isinstance(candidate, np.ndarray) else candidate
                subset = np.array(subset) if not isinstance(subset, np.ndarray) else subset
                if candidate.size > 0:
                    canvas = draw_bodypose(canvas, candidate, subset)
        
        # Draw faces (filter out invalid points near origin)
        if pose_dict.get("faces") is not None:
            faces = pose_dict["faces"]
            faces = np.array(faces) if not isinstance(faces, np.ndarray) else faces
            if faces.size > 0:
                # Filter out points that are essentially at origin (invalid)
                # Set very small coordinates to -1 (which draw_facepose ignores)
                mask = (faces[..., 0] < 5) & (faces[..., 1] < 5)
                faces[mask] = -1
                canvas = draw_facepose(canvas, faces)
        
        # Draw hands (filter out invalid points near origin)
        if pose_dict.get("hands") is not None:
            hands = pose_dict["hands"]
            hands = np.array(hands) if not isinstance(hands, np.ndarray) else hands
            if hands.size > 0:
                # Filter out points that are essentially at origin (invalid)
                mask = (hands[..., 0] < 5) & (hands[..., 1] < 5)
                hands[mask] = -1
                canvas = draw_handpose(canvas, hands)
        
        return canvas
        
    except ImportError:
        # Fallback to simple drawing if controlnet_aux is not available
        print("[T2P] Warning: controlnet_aux not found, using simple pose drawing")
        return draw_pose_simple(pose_dict, height, width)


def draw_pose_simple(pose_dict, height, width):
    """
    Simple fallback pose drawing without controlnet_aux dependency.
    Draws body keypoints and connections.
    """
    from PIL import Image, ImageDraw
    
    canvas = Image.new('RGB', (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    
    bodies = pose_dict.get("bodies", {})
    candidate = bodies.get("candidate")
    subset = bodies.get("subset")
    
    if candidate is None or subset is None:
        return np.array(canvas)
    
    candidate = np.array(candidate) if not isinstance(candidate, np.ndarray) else candidate
    subset = np.array(subset) if not isinstance(subset, np.ndarray) else subset
    
    # Draw each person
    for person_idx in range(len(subset)):
        person = subset[person_idx]
        
        # Draw limb connections
        for i, (start_idx, end_idx) in enumerate(LIMB_CONNECTIONS):
            if start_idx >= len(person) or end_idx >= len(person):
                continue
                
            start_kpt_idx = int(person[start_idx])
            end_kpt_idx = int(person[end_idx])
            
            if start_kpt_idx < 0 or end_kpt_idx < 0:
                continue
            if start_kpt_idx >= len(candidate) or end_kpt_idx >= len(candidate):
                continue
                
            start_pt = candidate[start_kpt_idx]
            end_pt = candidate[end_kpt_idx]
            
            # Skip invalid points
            if start_pt[0] < 0 or start_pt[1] < 0 or end_pt[0] < 0 or end_pt[1] < 0:
                continue
            
            color = BODY_COLORS[i % len(BODY_COLORS)]
            draw.line(
                [(int(start_pt[0]), int(start_pt[1])), 
                 (int(end_pt[0]), int(end_pt[1]))],
                fill=color,
                width=4
            )
        
        # Draw keypoints
        for kpt_idx in range(min(18, len(person))):
            candidate_idx = int(person[kpt_idx])
            if candidate_idx < 0 or candidate_idx >= len(candidate):
                continue
                
            pt = candidate[candidate_idx]
            if pt[0] < 0 or pt[1] < 0:
                continue
                
            color = BODY_COLORS[kpt_idx % len(BODY_COLORS)]
            x, y = int(pt[0]), int(pt[1])
            draw.ellipse(
                [(x - 4, y - 4), (x + 4, y + 4)],
                fill=color
            )
    
    # Draw hands (simplified)
    hands = pose_dict.get("hands")
    if hands is not None:
        hands = np.array(hands) if not isinstance(hands, np.ndarray) else hands
        for hand in hands:
            if hand is None or len(hand) == 0:
                continue
            for pt in hand:
                if pt[0] > 0 and pt[1] > 0:
                    x, y = int(pt[0]), int(pt[1])
                    draw.ellipse(
                        [(x - 2, y - 2), (x + 2, y + 2)],
                        fill=(255, 255, 255)
                    )
    
    # Draw faces (simplified)
    faces = pose_dict.get("faces")
    if faces is not None:
        faces = np.array(faces) if not isinstance(faces, np.ndarray) else faces
        for face in faces:
            if face is None or len(face) == 0:
                continue
            for pt in face:
                if pt[0] > 0 and pt[1] > 0:
                    x, y = int(pt[0]), int(pt[1])
                    draw.ellipse(
                        [(x - 1, y - 1), (x + 1, y + 1)],
                        fill=(255, 255, 255)
                    )
    
    return np.array(canvas)


def pose_to_openpose_format(pose_keypoints, width, height):
    """
    Convert internal pose keypoints to OpenPose JSON format.
    Useful for compatibility with other tools.
    """
    bodies = pose_keypoints.get("bodies", {})
    candidate = bodies.get("candidate", [])
    subset = bodies.get("subset", [])
    
    people = []
    for person in subset:
        keypoints = []
        for i in range(18):
            if i < len(person):
                idx = int(person[i])
                if idx >= 0 and idx < len(candidate):
                    pt = candidate[idx]
                    keypoints.extend([pt[0], pt[1], 1.0])  # x, y, confidence
                else:
                    keypoints.extend([0, 0, 0])
            else:
                keypoints.extend([0, 0, 0])
        
        people.append({
            "pose_keypoints_2d": keypoints
        })
    
    return {
        "version": 1.0,
        "people": people,
        "canvas_width": width,
        "canvas_height": height,
    }
