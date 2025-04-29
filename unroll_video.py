import cv2
import numpy as np
import argparse
import os
from tqdm import tqdm

def correct_rolling_shutter_video(input_path, direction='vertical', inverted=False, 
                                 strength=1.0, output_suffix='_corrected'):
    """
    Corrects rolling shutter deformation in videos.
    
    Parameters:
    -----------
    input_path : str
        Path to the input video file
    direction : str
        Direction of rolling shutter effect: 'vertical' or 'horizontal'
    inverted : bool
        Whether the rolling shutter direction is inverted
    strength : float
        Strength of the correction (0.0 to 2.0 recommended)
    output_suffix : str
        Suffix to add to the output filename
        
    Returns:
    --------
    str
        Path to the corrected video file
    """
    # Open the input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {input_path}")
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create output path
    filename, extension = os.path.splitext(input_path)
    output_path = f"{filename}{output_suffix}{extension}"
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You may need to adjust this codec
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process each frame
    print(f"Processing {frame_count} frames...")
    for _ in tqdm(range(frame_count)):
        ret, frame = cap.read()
        if not ret:
            break
            
        # Create output frame
        corrected = np.zeros_like(frame)
        
        # Apply correction based on direction
        if direction == 'vertical':
            # Vertical rolling shutter (rows captured at different times)
            for y in range(height):
                # Calculate shift amount based on row position
                progress = y / height  # 0.0 at top, 1.0 at bottom
                if inverted:
                    progress = 1.0 - progress
                    
                # Apply horizontal shift proportional to row position
                shift = int(strength * (progress - 0.5) * width * 0.1)
                
                # Apply the shift using affine transformation on this row
                M = np.float32([[1, 0, shift], [0, 1, 0]])
                row = frame[y:y+1, :]
                transformed_row = cv2.warpAffine(row, M, (width, 1))
                corrected[y:y+1, :] = transformed_row
                
        elif direction == 'horizontal':
            # Horizontal rolling shutter (columns captured at different times)
            for x in range(width):
                # Calculate shift amount based on column position
                progress = x / width  # 0.0 at left, 1.0 at right
                if inverted:
                    progress = 1.0 - progress
                    
                # Apply vertical shift proportional to column position
                shift = int(strength * (progress - 0.5) * height * 0.1)
                
                # Apply the shift using affine transformation on this column
                M = np.float32([[1, 0, 0], [0, 1, shift]])
                col = frame[:, x:x+1]
                transformed_col = cv2.warpAffine(col, M, (1, height))
                corrected[:, x:x+1] = transformed_col
        
        # Write corrected frame
        out.write(corrected)
    
    # Release resources
    cap.release()
    out.release()
    
    print(f"Corrected video saved to: {output_path}")
    return output_path

def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description='Correct rolling shutter deformation in videos')
    parser.add_argument('input_path', help='Path to the input video file')
    parser.add_argument('--direction', choices=['vertical', 'horizontal'], default='vertical',
                        help='Direction of rolling shutter effect')
    parser.add_argument('--inverted', action='store_true', 
                        help='Whether the rolling shutter direction is inverted')
    parser.add_argument('--strength', type=float, default=1.0,
                        help='Strength of the correction (0.0 to 2.0 recommended)')
    parser.add_argument('--output-suffix', default='_corrected',
                        help='Suffix to add to the output filename')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Apply correction
    try:
        output_path = correct_rolling_shutter_video(
            args.input_path,
            args.direction,
            args.inverted,
            args.strength,
            args.output_suffix
        )
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
